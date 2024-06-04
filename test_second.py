import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

from utils.evaluate import evaluate_prediction, pred_metrics, gt_metrics, all_metrics, metrics_coeffs
from utils.smplBody import BodyModel
from diffusion_stage.parser_util import get_args, merge_file
from dataloader.dataloader import load_data, TestDataset
from VQVAE.transformer_vqvae import TransformerVQVAE
from diffusion_stage.wrap_model import MotionDiffusion

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#####################
lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def overlapping_test_simplify(args, data, dataset, diff_model_upper, diff_model_lower, vq_model_upper, vq_model_lower,
                              num_per_batch=256):
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
        fullbody_res = torch.zeros((new_batch_size, new_seq, 22, 6)).to(device)
        with torch.no_grad():
            upper_latents = diff_model_upper.diffusion_reverse(sparse_per_batch.reshape(new_batch_size, new_seq, 3, 18))
            lower_latents = diff_model_lower.diffusion_reverse(sparse_per_batch.reshape(new_batch_size, new_seq, 3, 18),
                                                               upper_latents)
            upper_mat = vq_model_upper.decode_my(upper_latents, new_batch_size, args.INPUT_MOTION_LENGTH)
            lower_mat = vq_model_lower.decode_my(lower_latents, new_batch_size, args.INPUT_MOTION_LENGTH)
        upper_mat = upper_mat.reshape(new_batch_size, new_seq, len(upper_body), 6)
        lower_mat = lower_mat.reshape(new_batch_size, new_seq, len(lower_body), 6)
        fullbody_res[:, :, upper_body] = upper_mat
        fullbody_res[:, :, lower_body] = lower_mat

        sample = fullbody_res[:, -1].reshape(-1, 22 * 6)
        # sample = utils_transform.absSixd2rel_pavis_seq(sample)  # (seq, 132)
        # sample = sample.reshape(-1, 132)
        output_samples.append(sample.cpu().float())
    return output_samples, body_param, head_motion, filename, firstCounter, secondCounter


def test_process(args=None, log_path=None, cur_epoch=None):
    if args is None:
        cfg_args = get_args()
        # cfg_args.cfg = 'config_diffusion/second_low_rerun.yaml'
        cfg_args.cfg = 'config_diffusion/second.yaml'
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

    vqcfg = args.VQVAE
    vq_model_upper = TransformerVQVAE(in_dim=len(upper_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim,
                                      heads=vqcfg.heads, dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook,
                                      n_e=vqcfg.n_e, e_dim=vqcfg.e_dim, beta=vqcfg.beta)
    vq_model_lower = TransformerVQVAE(in_dim=len(lower_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim,
                                      heads=vqcfg.heads, dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook,
                                      n_e=vqcfg.n_e, e_dim=vqcfg.e_dim, beta=vqcfg.beta)
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

    diff_model_upper = MotionDiffusion(cfg=args.DIFFUSION, input_length=args.INPUT_MOTION_LENGTH,
                                       num_layers=args.DIFFUSION.layers_upper, use_upper=False)
    diff_model_lower = MotionDiffusion(cfg=args.DIFFUSION, input_length=args.INPUT_MOTION_LENGTH,
                                       num_layers=args.DIFFUSION.layers_lower, use_upper=True)
    upper_checkpoint_file = os.path.join(args.UPPER_DIF_DIR, 'best.pth.tar')
    if os.path.exists(upper_checkpoint_file):
        upper_checkpoint = torch.load(upper_checkpoint_file, map_location=lambda storage, loc: storage)
        diff_model_upper.load_state_dict(upper_checkpoint)
        print(f"loading upper diffusion model: {upper_checkpoint_file}")
    else:
        print(f"Upper Diffusion Model {upper_checkpoint_file} not exists!")
        return

    output_dir = args.SAVE_DIR
    model_file = os.path.join(output_dir, 'best.pth.tar')
    if os.path.exists(model_file):
        print(f"loading lower diffusion model {model_file}")
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        # diff_model_lower.load_state_dict(checkpoint['state_dict'])
        diff_model_lower.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Lower Diffusion Model {model_file} not exists!")
        # return

    vq_model_upper = vq_model_upper.to(device)
    vq_model_lower = vq_model_lower.to(device)
    diff_model_upper = diff_model_upper.to(device)
    diff_model_lower = diff_model_lower.to(device)
    vq_model_upper.eval()
    vq_model_lower.eval()
    diff_model_upper.eval()
    diff_model_lower.eval()

    if args.VIS:
        test_loader = range(len(dataset))
    else:
        test_loader = tqdm(range(len(dataset)))

    fC_all = Counter()
    sC_all = Counter()
    for sample_index in test_loader:
        # if dataset[sample_index][-1] not in ["CMU-14"]:
        #     continue
        output, body_param, head_motion, filename, fC, sC = (
            overlapping_test_simplify(args, dataset[sample_index], dataset, diff_model_upper, diff_model_lower,
                                      vq_model_upper, vq_model_lower, 1024))
        fC_all += fC
        sC_all += sC
        sample = torch.cat(output, dim=0)  # (N, 132)

        instance_log = evaluate_prediction(
            args, all_metrics, sample, body_model,
            head_motion, body_param, fps, filename
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
