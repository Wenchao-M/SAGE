import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
import torch.nn.functional as F
from test_refiner import test_process
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
    global_pos = root_trans[:, None] + pred_loc  # (bs, 22, 3)

    fk_loss = torch.mean(
        torch.norm((pred_loc - gt_loc).reshape(-1, 3), p=2, dim=1)
    )
    pred_jitter = (
        ((pred_loc[3:] - 3 * pred_loc[2:-1] +
          3 * pred_loc[1:-2] - pred_loc[:-3])).norm(dim=2).mean()
    )
    root_trans_gt = - gt_loc[:, 15] + gt_head_pos
    global_pos_gt = gt_loc + root_trans_gt[:, None]

    # whole_align_loss = torch.mean(
    #     torch.norm((global_pos - global_pos_gt).reshape(-1, 3), p=2, dim=1)
    # )
    align_loss = F.smooth_l1_loss(global_pos[:, [15, 20, 21]], global_pos_gt[:, [15, 20, 21]])
    return pred_jitter, fk_loss, align_loss


def do_train(args, diff_model_upper, diff_model_lower, vq_model_upper, vq_model_lower, decoder_model, refinenet,
             dataloader, log_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator(mixed_precision='fp16')

    support_dir = 'body_models'
    body_model = BodyModel(support_dir).to(device)

    upper_vq_dir = args.UPPER_VQ_DIR
    vqvae_upper_file = os.path.join(upper_vq_dir, 'best.pth.tar')
    if os.path.exists(vqvae_upper_file):
        checkpoint_upper = torch.load(vqvae_upper_file, map_location=lambda storage, loc: storage)
        vq_model_upper.load_state_dict(checkpoint_upper)
        print(f"=> Load upper vqvae {vqvae_upper_file}")
    else:
        print("No upper vqvae model!")
        return

    lower_vq_dir = args.LOWER_VQ_DIR
    vqvae_lower_file = os.path.join(lower_vq_dir, 'best.pth.tar')
    if os.path.exists(vqvae_lower_file):
        checkpoint_lower = torch.load(vqvae_lower_file, map_location=lambda storage, loc: storage)
        vq_model_lower.load_state_dict(checkpoint_lower)
        print(f"=> Load upper vqvae {vqvae_lower_file}")
    else:
        print("No lower vqvae model!")
        return

    decoder_file = args.FINETUNE_DIR
    decoder_weight = os.path.join(decoder_file, 'best.pth.tar')
    if os.path.exists(decoder_weight):
        checkpoint_all = torch.load(decoder_weight, map_location=lambda storage, loc: storage)
        diff_model_upper.load_state_dict(checkpoint_all['upper_state_dict'])
        diff_model_lower.load_state_dict(checkpoint_all['lower_state_dict'], strict=False)
        decoder_model.load_state_dict(checkpoint_all['decoder_state_dict'], strict=False)
        print("=> loading checkpoint '{}'".format(decoder_weight))
    else:
        print("=> No prev decoder model weight!")
        # load upper body diffusion model
        upper_checkpoint_file = os.path.join(args.UPPER_DIF_DIR, 'best.pth.tar')
        if os.path.exists(upper_checkpoint_file):
            upper_checkpoint = torch.load(upper_checkpoint_file, map_location=lambda storage, loc: storage)
            diff_model_upper.load_state_dict(upper_checkpoint)
            print(f"loading upper diffusion model: {upper_checkpoint_file}")
        else:
            print(f"Upper Diffusion Model {upper_checkpoint_file} not exists!")
            return

        # load lower body diffusion model
        lower_checkpoint_file = os.path.join(args.LOWER_DIF_DIR, 'best.pth.tar')
        if os.path.exists(lower_checkpoint_file):
            lower_checkpoint = torch.load(lower_checkpoint_file, map_location=lambda storage, loc: storage)
            diff_model_lower.load_state_dict(lower_checkpoint, strict=False)
            print(f"loading lower diffusion model: {lower_checkpoint_file}")
        else:
            print(f"Upper Diffusion Model {lower_checkpoint_file} not exists!")
            return

        # load decoder model
        dec_file_wo_finetune = args.DECODER_DIR
        dec_weight_wo_finetune = os.path.join(dec_file_wo_finetune, 'best.pth.tar')
        if os.path.exists(dec_weight_wo_finetune):
            decoder_model.load_state_dict(torch.load(dec_weight_wo_finetune), strict=False)
        else:
            return

    optimizer = AdamW(refinenet.parameters(), lr=args.LR, weight_decay=args.WEIGHT_DECAY)
    begin_epoch = 0
    output_dir = args.SAVE_DIR
    refiner_model_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    if os.path.exists(refiner_model_file):
        refiner_checkpoint = torch.load(refiner_model_file)
        begin_epoch = refiner_checkpoint['epoch']
        optimizer.load_state_dict(refiner_checkpoint['optimizer'])
        refinenet.load_state_dict(refiner_checkpoint['state_dict'])
    else:
        print("no refiner checkpoint!")

    vq_model_upper.eval()
    vq_model_lower.eval()
    diff_model_upper.eval()
    diff_model_lower.eval()
    decoder_model.eval()
    refinenet.train()

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.MILESTONES, args.GAMMA,
                                                        last_epoch=begin_epoch if begin_epoch else -1)
    (diff_model_upper, diff_model_lower, optimizer, dataloader, lr_scheduler,
     vq_model_upper, vq_model_lower, decoder_model, body_model, refinenet) = accelerator.prepare(
        diff_model_upper, diff_model_lower, optimizer, dataloader, lr_scheduler,
        vq_model_upper, vq_model_lower, decoder_model, body_model, refinenet)

    loss_weight = args.REFINER.loss_weight
    l1loss = torch.nn.SmoothL1Loss()
    for epoch in range(begin_epoch, 18):
        tqdm.write(f"Starting epoch {epoch}")
        tqdm.write(f"current_lr:{lr_scheduler.get_lr()[0]}")
        train_dataloader = tqdm(dataloader, dynamic_ncols=True)
        count_num = 0
        for motion_132, sparse in train_dataloader:
            motion_132 = motion_132[0].to(device)  # (batch, k, 132)
            sparse = sparse[0].to(device)  # (batch, k, 54)
            optimizer.zero_grad()
            # 通过vqvae将其encode成特征
            bs, seq = motion_132.shape[:2]
            with torch.no_grad():
                upper_latent_pred = diff_model_upper.diffusion_reverse(sparse.reshape(bs, seq, 3, 18))
                lower_latents_pred = diff_model_lower.diffusion_reverse(sparse.reshape(bs, seq, 3, 18),
                                                                        upper_latent_pred)
                recover_6d = decoder_model(upper_latent_pred, lower_latents_pred, sparse)

            recover_6d = recover_6d[:, -1]  # (seq, 132)
            recover_6d = recover_6d.reshape(1, -1, 132)
            pred_delta, _ = refinenet(recover_6d, None)
            pred_final = (recover_6d + pred_delta)
            recons_loss = l1loss(pred_final, motion_132[None, :, -1])

            pred_final = pred_final.squeeze(0)
            motion_gt = motion_132[:, -1]  # (seq-1, 132)
            vel_loss = l1loss(pred_final[1:] - pred_final[:-1], motion_gt[1:] - motion_gt[:-1])
            vel_loss2 = l1loss(pred_final[3::3] - pred_final[:-3:3], motion_gt[3::3] - motion_gt[:-3:3])

            pred_jitter, fk_loss, hand_align_loss = calc_fk_loss(body_model, pred_final, motion_gt,
                                                                 sparse[:, -1, :, 12:15])

            loss = (recons_loss * loss_weight.recons + vel_loss * loss_weight.vel_1 + vel_loss2 * loss_weight.vel_2
                    + fk_loss * loss_weight.fk_loss + hand_align_loss * loss_weight.hand_align + pred_jitter * loss_weight.jitter)
            accelerator.backward(loss)
            train_dataloader.set_description(f"e:{epoch},rec:{recons_loss:.2e},"
                                             f"vel:{vel_loss:.2e},fk:{fk_loss:.2e},jitter:{pred_jitter:.2e}")
            if count_num % 300 == 0 or count_num == len(train_dataloader) - 1:
                with open(log_path, 'a') as f:
                    f.write(
                        f"epoch:{epoch}, recons_loss:{recons_loss:.2e}, vel1_loss:{vel_loss:.2e}, vel2_loss:{vel_loss2:.2e},"
                        f"fk_loss:{fk_loss:.2e} hand_align_loss:{hand_align_loss:.2e} jitter_loss:{pred_jitter:.2e}\n")
            count_num += 1
            optimizer.step()

        lr_scheduler.step()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': refinenet.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, output_dir)
        save_best({
            'epoch': epoch + 1,
            'state_dict': refinenet.state_dict(),
        }, output_dir)
        test_process(args, log_path, epoch)
        train_dataloader.close()


def save_checkpoint(states, output_dir):
    checkpoint_file = os.path.join(output_dir, "checkpoint.pth.tar")
    torch.save(states, checkpoint_file)


def save_best(states, output_dir):
    best_file = os.path.join(output_dir, "best.pth.tar")
    torch.save(states, best_file)
