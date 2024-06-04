import os
import math
import random
import numpy as np
import torch
from tqdm import tqdm
from utils import utils_transform
from utils.metrics import get_metric_function

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from VQVAE.parser_util import get_args
from dataloader.dataloader import load_data, TestDataset
from VQVAE.transformer_vqvae import TransformerVQVAE
from utils.smplBody import BodyModel

lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
device = "cuda" if torch.cuda.is_available() else "cpu"

#####################
RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre",
    "upperre",
    "lowerre",
    "rootre",
    "mpjpe",
    "mpjve",
    "handpe",
    "upperpe",
    "lowerpe",
    "rootpe",
    "pred_jitter",
]
gt_metrics = [
    "gt_jitter",
]
all_metrics = pred_metrics + gt_metrics

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)  # 57.2958 grads
metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "upperre": RADIANS_TO_DEGREES,
    "lowerre": RADIANS_TO_DEGREES,
    "rootre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
    "gt_mpjpe": METERS_TO_CENTIMETERS,
    "gt_mpjve": METERS_TO_CENTIMETERS,
    "gt_handpe": METERS_TO_CENTIMETERS,
    "gt_rootpe": METERS_TO_CENTIMETERS,
    "gt_upperpe": METERS_TO_CENTIMETERS,
    "gt_lowerpe": METERS_TO_CENTIMETERS,
}


def overlapping_test_simplify(args, data, model, body_part, num_per_batch=256):
    gt_data, sparse_original, body_param, head_motion, filename = (data[0], data[1], data[2], data[3], data[4])
    num_frames = head_motion.shape[0]
    gt_data = gt_data.cuda().float()  # (seq, 132)
    sparse = sparse_original.cuda().float().reshape(num_frames, 54)
    head_motion = head_motion.cuda().float()

    gt_data_splits = []
    sparse_splits = []
    block_seq = args.INPUT_MOTION_LENGTH  # 32
    seq_pad = gt_data[:1].repeat(block_seq - 1, 1)
    sparse_pad = sparse[:1].repeat(block_seq - 1, 1)
    gt_data_pad = torch.cat((seq_pad, gt_data), dim=0)  # (31+seq, 396)
    sparse_pad = torch.cat((sparse_pad, sparse), dim=0)

    for i in range(num_frames):
        gt_data_splits.append(gt_data_pad[i: i + block_seq])
        sparse_splits.append(sparse_pad[i: i + block_seq])

    gt_data_splits = torch.stack(gt_data_splits)  # (x, 32, 396)
    sparse_splits = torch.stack(sparse_splits)

    n_steps = gt_data_splits.shape[0] // num_per_batch
    if len(gt_data_splits) % num_per_batch > 0:
        n_steps += 1

    output_samples = []
    num_joints = len(body_part)

    for step_index in range(n_steps):
        gt_per_batch = gt_data_splits[step_index * num_per_batch: (step_index + 1) * num_per_batch].to(device)
        sparse_per_batch = sparse_splits[step_index * num_per_batch: (step_index + 1) * num_per_batch].to(device)
        with torch.no_grad():
            bs, seq = gt_per_batch.shape[:2]
            gt_per_batch = gt_per_batch.reshape((bs, seq, -1, 6))
            gt_per_batch = gt_per_batch[:, :, body_part, :].reshape((bs, seq, -1))
            sample, _, indices = model(x=gt_per_batch, sparse=sparse_per_batch)
        sample = sample[:, -1].reshape(-1, num_joints * 6)
        # sample = utils_transform.absSixd2rel_pavis_seq(sample)  # (seq, 132)
        output_samples.append(sample.cpu().float())
    # gt_data2 = utils_transform.absSixd2rel_pavis_seq(gt_data[0])
    return output_samples, body_param, head_motion, filename


def evaluate_prediction(args, metrics, sample, body_model, head_motion, body_part, body_param, fps, filename):
    seq = sample.shape[0]
    motion_pred = sample.squeeze().cuda()  # (N, 132)
    # Get the  prediction from the model
    model_rot_input = (  # (N, 66)
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach()).reshape(motion_pred.shape[0], -1).float()
    )
    for k, v in body_param.items():
        body_param[k] = v.squeeze().cuda()
        body_param[k] = body_param[k][-model_rot_input.shape[0]:, ...]

    T_head2world = head_motion.clone().cuda()
    t_head2world = T_head2world[:, :3, 3].clone()
    # Get the offset between the head and other joints using forward kinematic model
    # body_pose_local: (batch, 52, 3) joints loction

    pred_temp = torch.zeros((seq, 22, 3), device="cuda")
    # gt_temp = torch.zeros((seq, 22, 3), device="cuda")
    pred_temp[:, body_part] = model_rot_input.reshape((seq, -1, 3))

    pred_temp = pred_temp.reshape((seq, -1))
    body_pose_local = body_model(
        {
            "pose_body": pred_temp[..., 3:],
            "root_orient": pred_temp[..., :3],
            # "root_orient": body_param["root_orient"]
        }
    ).Jtr

    # Get the offset in global coordiante system between head and body_world.
    t_head2root = -body_pose_local[:, 15, :]  # root - head location
    t_root2world = t_head2root + t_head2world.cuda()
    if len(body_part) == len(upper_body):
        predicted_body = body_model(
            {
                "pose_body": pred_temp[..., 3:],
                "root_orient": pred_temp[..., :3],
                # "root_orient": body_param["root_orient"],
                "trans": t_root2world,
            }
        )
    elif len(body_part) == len(lower_body):
        predicted_body = body_model(
            {
                "pose_body": pred_temp[..., 3:],
                "root_orient": pred_temp[..., :3],
            }
        )
    else:
        return

    predicted_position = predicted_body.Jtr[:, :22, :]
    # Get the predicted position and rotation

    # Get the  ground truth position from the model
    gt_pose = torch.cat((body_param["root_orient"], body_param["pose_body"]), dim=-1).reshape((seq, -1, 3))
    gt_pose_temp = torch.zeros((seq, 22, 3), device="cuda")
    gt_pose_temp[:, body_part, :] = gt_pose[:, body_part, :]
    gt_pose_temp = gt_pose_temp.reshape((seq, -1))
    if len(body_part) == len(upper_body):
        gt_body = body_model({
            "pose_body": gt_pose_temp[..., 3:],
            "root_orient": gt_pose_temp[..., :3],
            "trans": body_param["trans"]
        })
    elif len(body_part) == len(lower_body):
        gt_body = body_model({
            "pose_body": gt_pose_temp[..., 3:],
            "root_orient": gt_pose_temp[..., :3],
        })
    else:
        return

    gt_position = gt_body.Jtr[:, :22, :]
    gt_root_angle = body_param["root_orient"]
    predicted_root_angle = pred_temp[..., :3]
    # upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    # lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    eval_log = {}
    for metric in metrics:
        eval_log[metric] = (
            get_metric_function(metric)(
                predicted_position,
                pred_temp,
                predicted_root_angle,
                gt_position,
                gt_pose_temp,
                gt_root_angle,
                upper_body,
                lower_body,
                fps,
            ).cpu().numpy()
        )

    torch.cuda.empty_cache()
    return eval_log


def test_process():
    args = get_args()
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
        protocol=args.PROTOCOL,
        input_motion_length=args.INPUT_MOTION_LENGTH,
    )
    dataset = TestDataset(all_info, filename_list)

    log = {}
    for metric in all_metrics:
        log[metric] = 0

    body_part_name = args.part
    if body_part_name == "upper":
        body_part = upper_body
    elif body_part_name == "lower":
        body_part = lower_body
    else:
        print("Fail to recognize the body part name.")
        return
    in_dim = len(body_part) * 6

    vqcfg = args.VQVAE
    model = TransformerVQVAE(in_dim=in_dim, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim, heads=vqcfg.heads,
                             dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook, n_e=vqcfg.n_e, e_dim=vqcfg.e_dim,
                             beta=vqcfg.beta)
    model = model.to(device)
    model.eval()

    output_dir = args.SAVE_DIR
    model_file = os.path.join(output_dir, 'best.pth.tar')
    if os.path.exists(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
    else:
        print(f"{model_file} not exist!!!")
        # return

    n_testframe = args.NUM_PER_BATCH
    for sample_index in tqdm(range(len(dataset))):
        output, body_param, head_motion, filename = \
            overlapping_test_simplify(args, dataset[sample_index], model, body_part, n_testframe)
        sample = torch.cat(output, dim=0)  # (N, 132) N表示帧数
        instance_log = evaluate_prediction(
            args, all_metrics, sample, body_model, head_motion,
            body_part, body_param, fps, filename)
        for key in instance_log:
            log[key] += instance_log[key]
    # Print the value for all the metrics
    print("Metrics for the predictions")
    for metric in pred_metrics:
        print(metric, log[metric] / len(dataset) * metrics_coeffs[metric])
    print("Metrics for the ground truth")
    for metric in gt_metrics:
        print(metric, log[metric] / len(dataset) * metrics_coeffs[metric])


if __name__ == "__main__":
    test_process()
