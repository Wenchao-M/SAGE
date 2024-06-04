import os
import random
import numpy as np
import torch
import time
import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from diffusion_stage.parser_util import get_args, merge_file
from dataloader.dataloader import get_dataloader, load_data, TrainDataset
from VQVAE.transformer_vqvae import TransformerVQVAE
from diffusion_stage.wrap_model import MotionDiffusion
from diffusion_stage.do_train_decoder import do_train
from diffusion_stage.transformer_decoder import TransformerDecoder

lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def main():
    cfg_args = get_args()
    cfg_args.cfg = 'config_decoder/decoder.yaml'
    args = merge_file(cfg_args)
    name = cfg_args.cfg.split('/')[-1].split('.')[0]  # 输出保存地址
    args.SAVE_DIR = os.path.join("outputs_new", name)

    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)

    if args.SAVE_DIR is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)
    print(f"Saving to:{args.SAVE_DIR}")

    # 将args写到输出里去
    output_dir = args.SAVE_DIR
    timestamp = time.time()
    dt = datetime.datetime.fromtimestamp(timestamp)
    formatted_dt = dt.strftime("%Y%m%d_%H%M")
    log_path = os.path.join(output_dir, f"{name}_{formatted_dt}.log")
    with open(log_path, 'w') as f:
        f.write(str(args) + '\n')
        print(f"Args saving to {log_path}")

    print("creating data loader...")
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
        train_dataset_repeat_times=args.TRAIN_DATASET_REPEAT_TIMES
    )
    train_dataloader = get_dataloader(
        train_dataset, "train", batch_size=args.BATCH_SIZE, num_workers=args.NUM_WORKERS
    )
    print("creating model...")

    print("creating model and diffusion...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    args.num_workers = args.NUM_WORKERS * num_gpus

    vqcfg = args.VQVAE
    vq_model_upper = TransformerVQVAE(in_dim=len(upper_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim,
                                      heads=vqcfg.heads, dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook,
                                      n_e=vqcfg.n_e, e_dim=vqcfg.e_dim, beta=vqcfg.beta)
    vq_model_lower = TransformerVQVAE(in_dim=len(lower_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim,
                                      heads=vqcfg.heads, dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook,
                                      n_e=vqcfg.n_e, e_dim=vqcfg.e_dim, beta=vqcfg.beta)

    diff_model_upper = MotionDiffusion(cfg=args.DIFFUSION, input_length=args.INPUT_MOTION_LENGTH,
                                       num_layers=args.DIFFUSION.layers_upper, use_upper=False)
    diff_model_lower = MotionDiffusion(cfg=args.DIFFUSION, input_length=args.INPUT_MOTION_LENGTH,
                                       num_layers=args.DIFFUSION.layers_lower, use_upper=True)
    decoder_model = TransformerDecoder(in_dim=132, seq_len=args.INPUT_MOTION_LENGTH, **args.DECODER)

    vq_model_upper = vq_model_upper.to(device)
    vq_model_lower = vq_model_lower.to(device)
    diff_model_upper = diff_model_upper.to(device)
    diff_model_lower = diff_model_lower.to(device)
    decoder_model = decoder_model.to(device)

    print("Training...")
    do_train(args, diff_model_upper, diff_model_lower, vq_model_upper,
             vq_model_lower, decoder_model, train_dataloader, log_path)
    print("Done.")


if __name__ == "__main__":
    main()
