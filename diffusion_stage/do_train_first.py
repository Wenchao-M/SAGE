import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
import torch.nn.functional as F
from test_first import test_process

lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
upper_body_cont = [1, 2, 4, 5, 7, 8, 10, 11]
lower_body_cont = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def do_train(args, model, vq_model_upper, dataloader, log_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator(mixed_precision='fp16')

    upper_vq_dir = args.UPPER_VQ_DIR
    vqvae_upper_file = os.path.join(upper_vq_dir, 'best.pth.tar')  # 目前用的root0.2的
    if os.path.exists(vqvae_upper_file):
        checkpoint_upper = torch.load(vqvae_upper_file, map_location=lambda storage, loc: storage)
        vq_model_upper.load_state_dict(checkpoint_upper)
        print(f"=> Load upper vqvae {vqvae_upper_file}")
    else:
        print("No vqvae model!")
        return

    begin_epoch = 0
    output_dir = args.SAVE_DIR
    checkpoint_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        min_mpjpe = checkpoint["min_mpjpe"]
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
        print(f"Current min MPJPE:{min_mpjpe}")
    else:
        min_mpjpe = 999
        print(f"{checkpoint_file} not exists, start training from scratch >_<")

    vq_model_upper.eval()
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.LR, weight_decay=args.WEIGHT_DECAY)
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        optimizer.load_state_dict(checkpoint['optimizer'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.MILESTONES, args.GAMMA,
                                                        last_epoch=begin_epoch if begin_epoch else -1)
    model, optimizer, dataloader, lr_scheduler, vq_model_upper = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler, vq_model_upper)

    for epoch in range(begin_epoch, 51):
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
            motion_upper = motion_copy[:, :, upper_body].reshape(bs, seq, -1)
            with torch.no_grad():
                latents_upper = vq_model_upper.encode_my(motion_upper, sparse)

            ori_upper_pred = model(latents_upper, sparse)
            # ori_upper_pred
            upper_loss = F.smooth_l1_loss(ori_upper_pred, latents_upper)
            accelerator.backward(upper_loss)
            if count_num % 300 == 0 or count_num == len(train_dataloader) - 1:
                with open(log_path, 'a') as f:
                    f.write(f"epoch:{epoch}, upper:{upper_loss:.2e}\n")
            count_num += 1
            train_dataloader.set_description(f"e:{epoch},upper:{upper_loss:.2e}")
            optimizer.step()

        lr_scheduler.step()

        # 保存模型
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_mpjpe': min_mpjpe,
        }, output_dir)
        save_best({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
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
