import copy
import os
import random
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import torch.multiprocessing as mp
from utils import utils_transform

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from VQVAE.parser_util import get_args
from dataloader.dataloader import get_dataloader, load_data, TrainDataset
from VQVAE.transformer_vqvae import TransformerVQVAE
from utils.smplBody import BodyModel
from test_vqvae import test_process

lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def loss_function(args, recover_6d, motion, loss_z, bodymodel, gt_pos, body_part):
    # import pdb;pdb.set_trace()
    loss_func = torch.nn.SmoothL1Loss(reduction='none')
    bs = recover_6d.shape[0]
    recover = recover_6d.reshape(bs, -1, 6)
    gt = motion.reshape(bs, -1, 6)
    pred_temp = torch.zeros((bs, 22, 3), device="cuda")
    gt_temp = torch.zeros((bs, 22, 3), device="cuda")
    pred_aa = utils_transform.sixd2aa(recover, batch=True)
    gt_aa = utils_transform.sixd2aa(gt, batch=True)
    pred_temp[:, body_part, :] = pred_aa
    gt_temp[:, body_part, :] = gt_aa
    pred_temp = pred_temp.flatten(1, 2)
    gt_temp = gt_temp.flatten(1, 2)

    pred_loc = bodymodel({
        "root_orient": pred_temp[:, :3],
        "pose_body": pred_temp[:, 3:]
    }).Jtr[:, :22, :]
    gt_loc = bodymodel({
        "root_orient": gt_temp[:, :3],
        "pose_body": gt_temp[:, 3:]
    }).Jtr[:, :22, :]

    gt_head_pos = gt_pos[:, 0]  # (bs, 3)
    head2root = pred_loc[:, 15].clone()  # (bs, 3)
    root_trans = gt_head_pos - head2root
    global_pos_pred = root_trans[:, None] + pred_loc  # (bs, 22, 3)

    if len(body_part) == len(upper_body):
        hand_align_loss = loss_func(global_pos_pred[:, [20, 21]], gt_pos[:, [1, 2]]).mean()
    elif len(body_part) == len(lower_body):
        hand_align_loss = 0
    else:
        print("Fail to recognize the body part!")
        return

    if args.ROOTLOSS:
        rec_root = loss_func(recover[:, 0], gt[:, 0]).mean()
        rec_other = loss_func(recover[:, 1:], gt[:, 1:]).mean()
        recons_loss = rec_root * 0.2 + rec_other
    else:
        recons_loss = loss_func(recover, gt).mean()

    fk_loss = loss_func(pred_loc, gt_loc)[:, body_part, :].mean()
    vq_loss = torch.mean(loss_z['quant_loss'])
    loss_w = args.LOSS
    loss_all = (recons_loss  + vq_loss * loss_w.alpha_codebook + fk_loss * loss_w.fk_loss +
                hand_align_loss * loss_w.hand_align_loss)
    loss = {
        "loss": loss_all,
        "vq_loss": vq_loss,
        "recons_loss": recons_loss,
        "fk_loss": fk_loss,
        "hand_align_loss": hand_align_loss
    }
    return loss


def save_checkpoint(states, output_dir):
    checkpoint_file = os.path.join(output_dir, "checkpoint.pth.tar")
    torch.save(states, checkpoint_file)
    if True:
        torch.save(
            states['state_dict'],
            os.path.join(output_dir, 'best.pth.tar')
        )


def do_train(args, model, train_dataloader, body_part):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    begin_epoch = 0
    output_dir = args.SAVE_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.LR, betas=(0.9, 0.99), weight_decay=args.WEIGHT_DECAY)
    if os.path.exists(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.MILESTONES, gamma=1 / 4, last_epoch=begin_epoch if begin_epoch else -1)

    support_dir = 'body_models'
    body_model = BodyModel(support_dir).to(device)
    model.train()

    num_joints = len(body_part)
    for epoch in range(begin_epoch, args.EPOCH):
        tqdm.write(f"Starting epoch {epoch}")
        tqdm.write(f"current lr:{scheduler.get_last_lr()}")
        train_dataloader = tqdm(train_dataloader, dynamic_ncols=True)
        for motion, sparse in train_dataloader:
            bs, seq = motion.shape[:2]
            motion = motion.to(device)
            sparse = sparse.to(device)
            motion_input = copy.deepcopy(motion)  # (bs, 20, 396)
            motion_input = motion_input.reshape(bs, seq, 22, 6)[:, :, body_part, :]
            motion = motion.reshape(bs, seq, 22, 6)[:, :, body_part, :]
            if True:
                random_mask = torch.rand((bs, num_joints))
                random_mask[:, 0] = 1
                random_mask = random_mask.unsqueeze(1).repeat(1, seq, 1)
                random_mask = torch.where(random_mask < args.MASK_RATIO)
                motion_input[random_mask] = 0.01

            recover_6d, loss_z, indices = model(x=motion_input, sparse=sparse)
            loss = loss_function(args, recover_6d[:, -1], motion[:, -1], loss_z,
                                 body_model, sparse[:, -1, :, 12:15], body_part)  # 以cm为单位
            optimizer.zero_grad()
            loss["loss"].backward()
            optimizer.step()
            train_dataloader.set_description(f"e:{epoch},rc:{loss['recons_loss']:.2e},vq:{loss['vq_loss']:.2e},"
                                             f"fk:{loss['fk_loss']:.2e},hd:{loss['hand_align_loss']:.2e}")

        scheduler.step()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, output_dir)

        test_process()
        train_dataloader.close()


def main():
    args = get_args()
    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    if args.SAVE_DIR is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)

    motions, sparses, all_info = load_data(
        args.DATASET_PATH,
        "train",
        protocol=args.PROTOCOL,
        input_motion_length=args.INPUT_MOTION_LENGTH,
    )
    train_dataset = TrainDataset(
        motions,
        sparses,
        all_info,
        args.INPUT_MOTION_LENGTH,
        args.TRAIN_DATASET_REPEAT_TIMES,
    )
    train_dataloader = get_dataloader(
        train_dataset, "train", batch_size=args.BATCH_SIZE, num_workers=args.NUM_WORKERS
    )

    print("creating model...")
    print(f"{args.SAVE_DIR}")
    body_part_name = args.part
    if body_part_name == "upper":
        body_part = upper_body
    elif body_part_name  == "lower":
        body_part =lower_body
    else:
        print("Fail to recognize the body part name.")
        return

    in_dim = len(body_part) * 6
    vqcfg = args.VQVAE
    model = TransformerVQVAE(in_dim=in_dim, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim, heads=vqcfg.heads,
                             dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook, n_e=vqcfg.n_e, e_dim=vqcfg.e_dim,
                             beta=vqcfg.beta)

    print("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")
    do_train(args, model, train_dataloader, body_part)
    print("Done.")


if __name__ == '__main__':
    mp.set_sharing_strategy('file_system')
    main()
