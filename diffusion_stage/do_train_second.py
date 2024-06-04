import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
import torch.nn.functional as F
from test_second import test_process

lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
upper_body_cont = [1, 2, 4, 5, 7, 8, 10, 11]
lower_body_cont = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def do_train(args, diff_model_upper, diff_model_lower, vq_model_upper, vq_model_lower, dataloader, log_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator(mixed_precision='fp16')

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

    upper_checkpoint_file = os.path.join(args.UPPER_DIF_DIR, 'best.pth.tar')
    if os.path.exists(upper_checkpoint_file):
        upper_checkpoint = torch.load(upper_checkpoint_file, map_location=lambda storage, loc: storage)
        diff_model_upper.load_state_dict(upper_checkpoint)
        print(f"loading upper diffusion model: {upper_checkpoint_file}")
    else:
        print(f"Upper Diffusion Model {upper_checkpoint_file} not exists!")
        return

    begin_epoch = 0
    output_dir = args.SAVE_DIR
    lower_checkpoint_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    if os.path.exists(lower_checkpoint_file):
        lower_checkpoint = torch.load(lower_checkpoint_file, map_location=lambda storage, loc: storage)
        begin_epoch = lower_checkpoint['epoch']
        min_mpjpe = lower_checkpoint["min_mpjpe"]
        diff_model_lower.load_state_dict(lower_checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(lower_checkpoint_file, lower_checkpoint['epoch']))
        print(f"Current min MPJPE:{min_mpjpe}")
    else:
        min_mpjpe = 999
        print(f"{lower_checkpoint_file} not exists, start training from scratch >_<")

    vq_model_upper.eval()
    vq_model_lower.eval()
    diff_model_upper.eval()
    diff_model_lower.train()

    optimizer = AdamW(diff_model_lower.parameters(), lr=args.LR, weight_decay=args.WEIGHT_DECAY)
    if os.path.exists(lower_checkpoint_file):
        lower_checkpoint = torch.load(lower_checkpoint_file, map_location=lambda storage, loc: storage)
        optimizer.load_state_dict(lower_checkpoint['optimizer'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.MILESTONES, args.GAMMA,
                                                        last_epoch=begin_epoch if begin_epoch else -1)
    diff_model_upper, diff_model_lower, optimizer, dataloader, lr_scheduler, vq_model_upper, vq_model_lower = accelerator.prepare(
        diff_model_upper, diff_model_lower, optimizer, dataloader, lr_scheduler, vq_model_upper, vq_model_lower)

    for epoch in range(begin_epoch, 41):
        tqdm.write(f"Starting epoch {epoch}")
        tqdm.write(f"current lr:{lr_scheduler.get_last_lr()}")
        train_dataloader = tqdm(dataloader, dynamic_ncols=True)
        count_num = 0
        for motion_132, sparse in train_dataloader:
            motion_132 = motion_132.to(device)  # (batch, k, 132)
            sparse = sparse.to(device)  # (batch, k, 54
            optimizer.zero_grad()
            bs, seq = motion_132.shape[:2]
            motion_copy = motion_132.reshape(bs, seq, 22, 6)
            motion_lower = motion_copy[:, :, lower_body].reshape(bs, seq, -1)
            with torch.no_grad():
                upper_latent_pred = diff_model_upper.diffusion_reverse(sparse)
                latents_lower_gt = vq_model_lower.encode_my(motion_lower, sparse)
            ori_lower_pred = diff_model_lower(latents_lower_gt, sparse, upper_latent_pred)

            lower_loss = F.smooth_l1_loss(ori_lower_pred, latents_lower_gt)
            accelerator.backward(lower_loss)
            if count_num % 300 == 0 or count_num == len(train_dataloader) - 1:
                with open(log_path, 'a') as f:
                    f.write(f"epoch:{epoch}, lower:{lower_loss:.2e}\n")
            count_num += 1
            train_dataloader.set_description(f"e:{epoch}, lower:{lower_loss:.2e}")
            optimizer.step()

        lr_scheduler.step()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': diff_model_lower.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_mpjpe': min_mpjpe,
        }, output_dir)
        save_best({
            'epoch': epoch + 1,
            'state_dict': diff_model_lower.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_mpjpe': min_mpjpe,
        }, output_dir)
        mpjpe = test_process(args, log_path, epoch)
        train_dataloader.close()


def save_checkpoint(states, output_dir):
    checkpoint_file = os.path.join(output_dir, "checkpoint.pth.tar")
    torch.save(states, checkpoint_file)


def save_best(states, output_dir):
    torch.save(
        states['state_dict'],
        os.path.join(output_dir, 'best.pth.tar')
    )
