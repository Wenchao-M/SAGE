import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

import utils.utils_transform
from utils.smplBody import BodyModel
from utils.evaluate import evaluate_prediction, pred_metrics, gt_metrics, all_metrics, metrics_coeffs
from diffusion_stage.parser_util import get_args, merge_file
from dataloader.dataloader import load_data, TestDataset
from VQVAE.transformer_vqvae import TransformerVQVAE
from diffusion_stage.wrap_model import MotionDiffusion

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def overlapping_test_simplify(args, data, dataset, model, vq_model_upper, num_per_batch=256):
    gt_data, sparse_original, body_param, head_motion, filename = (data[0], data[1], data[2], data[3], data[4])
    sparse_original = sparse_original.cuda().float()  # (seq, 54)
    head_motion = head_motion.cuda().float()
    num_frames = head_motion.shape[0]

    gt_data_splits = []
    block_seq = args.INPUT_MOTION_LENGTH  # 32
    seq_pad = sparse_original[:1]  # .repeat(args.INPUT_MOTION_LENGTH - 1, 1, 1)
    # seq_pad[..., 6:12] = torch.tensor([1., 0, 0, 0, 1, 0]).float().cuda().reshape(1, 1, 6).repeat(1, 3, 1)
    # seq_pad[..., 15:] = torch.tensor([0.0, 0, 0]).float().cuda().reshape(1, 1, 3).repeat(1, 3, 1)
    seq_pad = seq_pad.repeat(args.INPUT_MOTION_LENGTH - 1, 1, 1)
    gt_data_pad = torch.cat((seq_pad, sparse_original), dim=0)  # (31+seq, 54)

    for i in range(num_frames):
        gt_data_splits.append(gt_data_pad[i: i + block_seq])
    gt_data_splits = torch.stack(gt_data_splits)  # (x, 32, 54)

    n_steps = gt_data_splits.shape[0] // num_per_batch
    if len(gt_data_splits) % num_per_batch > 0:
        n_steps += 1

    firstCounter = Counter()
    secondCounter = Counter()
    output_samples = []
    for step_index in range(n_steps):
        sparse_per_batch = gt_data_splits[step_index * num_per_batch: (step_index + 1) * num_per_batch].to(device)
        new_batch_size, new_seq = sparse_per_batch.shape[:2]
        # fullbody_res = torch.zeros((new_batch_size, new_seq, 22, 6)).to(device)
        with torch.no_grad():
            upper_latents = model.diffusion_reverse(sparse_per_batch.reshape(new_batch_size, new_seq, 3, 18))
            upper_mat = vq_model_upper.decode_my(upper_latents, new_batch_size, args.INPUT_MOTION_LENGTH)
        upper_mat = upper_mat.reshape(new_batch_size, new_seq, len(upper_body), 6)
        sample = upper_mat[:, -1].reshape(-1, len(upper_body) * 6)
        output_samples.append(sample.cpu().float())
    return output_samples, body_param, head_motion, filename


def test_process(args=None, log_path=None, cur_epoch=None):
    if args is None:
        cfg_args = get_args()
        cfg_args.cfg = 'config_diffusion/first.yaml'
        args = merge_file(cfg_args)
        name = cfg_args.cfg.split('/')[-1].split('.')[0]
        args.SAVE_DIR = os.path.join("outputs", name)

    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)

    fps = args.FPS  # AMASS dataset requires 60 frames per second

    body_model = BodyModel(args.SUPPORT_DIR).to(device)
    print("Loading dataset...")
    filename_list, all_info = load_data(
        args.DATASET_PATH,
        "test",
        protocol=args.PROTOCOL
    )
    dataset = TestDataset(all_info, filename_list)

    log = {}
    for metric in all_metrics:
        log[metric] = 0

    body_part = upper_body
    vqcfg = args.VQVAE
    vq_model_upper = TransformerVQVAE(in_dim=len(body_part) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim,
                                      heads=vqcfg.heads, dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook,
                                      n_e=vqcfg.n_e, e_dim=vqcfg.e_dim, beta=vqcfg.beta)
    vq_dir = args.UPPER_VQ_DIR
    vqvae_upper_file = os.path.join(vq_dir, 'best.pth.tar')
    if os.path.exists(vqvae_upper_file):
        checkpoint_upper = torch.load(vqvae_upper_file, map_location=lambda storage, loc: storage)
        vq_model_upper.load_state_dict(checkpoint_upper)
        print(f"=> loaded vqvae {vqvae_upper_file}")
    else:
        print("No vqvae model!")
        return

    diff_model_upper = MotionDiffusion(cfg=args.DIFFUSION, input_length=args.INPUT_MOTION_LENGTH,
                                       num_layers=args.DIFFUSION.layers_upper, use_upper=False)
    output_dir = args.SAVE_DIR
    model_file = os.path.join(output_dir, 'best.pth.tar')
    if os.path.exists(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        diff_model_upper.load_state_dict(checkpoint)
    else:
        print(f"{model_file} not exist!!!")

    vq_model_upper = vq_model_upper.to(device)
    diff_model_upper = diff_model_upper.to(device)
    vq_model_upper.eval()
    diff_model_upper.eval()

    if args.VIS:
        test_loader = range(len(dataset))
    else:
        test_loader = tqdm(range(len(dataset)))

    for sample_index in test_loader:
        # if dataset[sample_index][-1] not in ["CMU-14"]:
        #     continue
        output, body_param, head_motion, filename = (
            overlapping_test_simplify(args, dataset[sample_index], dataset, diff_model_upper,
                                      vq_model_upper, 1024))
        sample = torch.cat(output, dim=0)

        seq_len = sample.shape[0]
        fullbody_res_aa = torch.zeros((seq_len * 22, 3)).to(sample.device)
        fullbody_res = utils.utils_transform.aa2sixd(fullbody_res_aa).reshape(seq_len, 22, 6)
        fullbody_res[:, upper_body] = sample.reshape(seq_len, len(upper_body), 6)
        fullbody_res = fullbody_res.reshape(seq_len, 132)
        lower_mask_idx = [0, 1, 3, 4, 6, 7, 9, 10]
        body_param['pose_body'].reshape(-1, 21, 3)[:, lower_mask_idx] *= 0.0

        instance_log = evaluate_prediction(
            args, all_metrics, fullbody_res, body_model,
            head_motion, body_param, fps, filename, "upper"
        )
        for key in instance_log:
            log[key] += instance_log[key]

    print("Metrics for the predictions")
    result_str = "\n"
    if cur_epoch is not None:
        result_str += f"epoch{cur_epoch} \n"
    for metric in pred_metrics:
        result_str += f"{metric}: {log[metric] / len(dataset) * metrics_coeffs[metric]} \n"
    for metric in gt_metrics:
        result_str += f"{metric}: {log[metric] / len(dataset) * metrics_coeffs[metric]} \n"
    print(result_str)
    if log_path is not None:
        with open(log_path, 'a') as f:
            f.write(result_str)
            print(f"Evalution results save to {log_path}")

    return log["mpjpe"] / len(dataset) * metrics_coeffs["mpjpe"]


if __name__ == "__main__":
    test_process()
