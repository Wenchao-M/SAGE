import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from test_decoder import test_process
from utils.smplBody import BodyModel
from utils import utils_transform

lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def calc_fk_loss(body_model, recover, gt, gt_pos):
    recover = recover.reshape(-1, 22, 6)
    gt = gt.reshape(-1, 22, 6)
    pred_aa = utils_transform.sixd2aa(recover, batch=True).flatten(1, 2)
    gt_aa = utils_transform.sixd2aa(gt, batch=True).flatten(1, 2)
    pred_loc = body_model({
        "root_orient": pred_aa[:, :3],
        "pose_body": pred_aa[:, 3:]
    }).Jtr[:, :22, :]
    gt_loc = body_model({
        "root_orient": gt_aa[:, :3],
        "pose_body": gt_aa[:, 3:]
    }).Jtr[:, :22, :]

    gt_head_pos = gt_pos[:, 0]  # (bs, 3)
    head2root = pred_loc[:, 15].clone()  # (bs, 3)
    root_trans = gt_head_pos - head2root
    global_pos_pred = root_trans[:, None] + pred_loc  # (bs, 22, 3)

    root_trans_gt = - gt_loc[:, 15] + gt_head_pos
    global_pos_gt = gt_loc + root_trans_gt[:, None]

    whole_align_loss = torch.mean(
        torch.norm((global_pos_pred - global_pos_gt).reshape(-1, 3), p=2, dim=1)
    )
    fk_loss = torch.mean(
        torch.norm((pred_loc - gt_loc).reshape(-1, 3), p=2, dim=1)
    )
    return fk_loss, whole_align_loss


def do_train(args, diff_model_upper, diff_model_lower, vq_model_upper, vq_model_lower, decoder_model, dataloader,
             log_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator(mixed_precision='fp16')

    support_dir = 'body_models'
    body_model = BodyModel(support_dir).to(device)

    upper_vq_dir = args.UPPER_VQ_DIR
    vqvae_upper_file = os.path.join(upper_vq_dir, 'best.pth.tar')
    if os.path.exists(vqvae_upper_file):
        checkpoint_upper = torch.load(vqvae_upper_file)
        vq_model_upper.load_state_dict(checkpoint_upper)
        print(f"=> Load upper vqvae {vqvae_upper_file}")
    else:
        print("No upper vqvae model!")
        return

    lower_vq_dir = args.LOWER_VQ_DIR
    vqvae_lower_file = os.path.join(lower_vq_dir, 'best.pth.tar')
    if os.path.exists(vqvae_lower_file):
        checkpoint_lower = torch.load(vqvae_lower_file)
        vq_model_lower.load_state_dict(checkpoint_lower)
        print(f"=> Load upper vqvae {vqvae_lower_file}")
    else:
        print("No lower vqvae model!")
        return

    output_dir = args.SAVE_DIR
    output_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    begin_epoch = 0
    if os.path.exists(output_file):
        checkpoint_all = torch.load(output_file)
        if checkpoint_all['epoch'] >= args.EPOCH_DECODER:
            stage = 'finetune'
            print("Finetune stage start!")
        else:
            stage = 'decoder'
            print("Decoder stage start!")
        begin_epoch = checkpoint_all['epoch']
        diff_model_upper.load_state_dict(checkpoint_all['upper_state_dict'])
        diff_model_lower.load_state_dict(checkpoint_all['lower_state_dict'])
        decoder_model.load_state_dict(checkpoint_all['decoder_state_dict'])
    else:
        stage = 'decoder'
        print("Start training from scratch!")
        upper_checkpoint_file = os.path.join(args.UPPER_DIF_DIR, 'best.pth.tar')
        if os.path.exists(upper_checkpoint_file):
            upper_checkpoint = torch.load(upper_checkpoint_file)
            diff_model_upper.load_state_dict(upper_checkpoint)
            print(f"loading upper diffusion model: {upper_checkpoint_file}")
        else:
            print(f"Upper Diffusion Model {upper_checkpoint_file} not exists!")
            return

        lower_checkpoint_file = os.path.join(args.LOWER_DIF_DIR, 'best.pth.tar')
        if os.path.exists(lower_checkpoint_file):
            lower_checkpoint = torch.load(lower_checkpoint_file)
            diff_model_lower.load_state_dict(lower_checkpoint)
            print(f"loading lower diffusion model: {lower_checkpoint_file}")
        else:
            print(f"Upper Diffusion Model {lower_checkpoint_file} not exists!")
            return

    vq_model_upper.eval()
    vq_model_lower.eval()
    decoder_model.train()
    if stage == 'decoder':
        diff_model_upper.eval()
        diff_model_lower.eval()
        optimizer = AdamW(decoder_model.parameters(), lr=args.LR_1, weight_decay=args.WEIGHT_DECAY)
    elif stage == 'finetune':
        diff_model_upper.train()
        diff_model_lower.train()
        optimizer = AdamW([{'params': diff_model_upper.parameters()}, {'params': diff_model_lower.parameters()},
                           {'params': decoder_model.parameters()}], lr=args.LR_2, weight_decay=args.WEIGHT_DECAY)
    else:
        print(f'Stage:{stage} not exists!')
        return

    if os.path.exists(output_file):
        decoder_checkpoint = torch.load(output_file)
        optimizer.load_state_dict(decoder_checkpoint['optimizer'])

    (diff_model_upper, diff_model_lower, optimizer, dataloader, vq_model_upper, vq_model_lower, decoder_model,
     body_model) = accelerator.prepare(diff_model_upper, diff_model_lower, optimizer, dataloader,
                                       vq_model_upper, vq_model_lower, decoder_model, body_model)

    loss_weight = args.DECODER.loss_weight
    opt_flag = True
    for epoch in range(begin_epoch, args.EPOCH_ALL):
        tqdm.write(f"Starting epoch {epoch}")
        tqdm.write(f"learning rate: {optimizer.param_groups[0]['lr']}")
        train_dataloader = tqdm(dataloader, dynamic_ncols=True)
        count_num = 0
        if epoch == args.EPOCH_DECODER:
            optimizer = AdamW(
                [{'params': diff_model_upper.parameters()}, {'params': diff_model_lower.parameters()},
                 {'params': decoder_model.parameters()}], lr=args.LR_2,
                weight_decay=args.WEIGHT_DECAY)
        for motion_132, sparse in train_dataloader:
            motion_132 = motion_132.to(device)  # (batch, k, 132)
            sparse = sparse.to(device)  # (batch, k, 54
            optimizer.zero_grad()
            if epoch < args.EPOCH_DECODER:
                with torch.no_grad():
                    upper_latent_pred = diff_model_upper.diffusion_reverse(sparse)
                    lower_latent_pred = diff_model_lower.diffusion_reverse(sparse, upper_latent_pred)
            else:
                upper_latent_pred = diff_model_upper.diffusion_reverse(sparse)
                lower_latent_pred = diff_model_lower.diffusion_reverse(sparse, upper_latent_pred)
            recover_6d = decoder_model(upper_latent_pred, lower_latent_pred, sparse)

            root_recons = torch.mean(
                torch.norm((recover_6d[:, -1, :6] - motion_132[:, -1, :6]).reshape(-1, 6), p=2, dim=1)
            )
            other_recons = torch.mean(
                torch.norm((recover_6d[:, -1, 6:] - motion_132[:, -1, 6:]).reshape(-1, 6), p=2, dim=1)
            )
            fk_loss, body_align_loss = calc_fk_loss(body_model, recover_6d[:, -1], motion_132[:, -1],
                                                    sparse[:, -1, :, 12:15])
            loss = (root_recons * loss_weight.root + other_recons * loss_weight.other_joints +
                    body_align_loss * loss_weight.body_fk)
            accelerator.backward(loss)
            if count_num % 300 == 0 or count_num == len(train_dataloader) - 1:
                with open(log_path, 'a') as f:
                    f.write(f"epoch:{epoch}, root_recons:{root_recons:.2e}, other_recons:{other_recons:.2e}, "
                            f"body_align_loss:{body_align_loss:.2e},fk_loss:{fk_loss:.2e}\n")
            count_num += 1
            train_dataloader.set_description(f"e:{epoch},r_rec:{root_recons:.2e},o_rec:{other_recons:.2e}, "
                                             f"hl:{body_align_loss:.2e},fk:{fk_loss:.2e}")
            optimizer.step()

        save_checkpoint({
            'epoch': epoch,
            'upper_state_dict': diff_model_upper.state_dict(),
            'lower_state_dict': diff_model_lower.state_dict(),
            'decoder_state_dict': decoder_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, output_dir)
        save_best({
            'epoch': epoch,
            'upper_state_dict': diff_model_upper.state_dict(),
            'lower_state_dict': diff_model_lower.state_dict(),
            'decoder_state_dict': decoder_model.state_dict(),
        }, output_dir)
        mpjpe = test_process(args, log_path, epoch)
        train_dataloader.close()


def save_checkpoint(states, output_dir):
    checkpoint_file = os.path.join(output_dir, "checkpoint.pth.tar")
    torch.save(states, checkpoint_file)


def save_best(states, output_dir):
    checkpoint_file = os.path.join(output_dir, "best.pth.tar")
    torch.save(states, checkpoint_file)
