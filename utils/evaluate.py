import torch
import os
import math
from utils import utils_visualize
from utils import utils_transform
from utils.metrics import get_metric_function

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0
pred_metrics = [
    "mpjre",
    "upperre",
    "lowerre",
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
# upper/lower_index are used to evaluate the results following AGRoL
upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
# upper_body_part is not the same as upper_index
upper_body_part = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
lower_body_part = [0, 1, 2, 4, 5, 7, 8, 10, 11]


def evaluate_prediction(args, metrics, sample, body_model, head_motion, body_param, fps, filename,
                        use_body_part="full"):
    motion_pred = sample.squeeze().cuda()  # (seq_len, 132)
    seq_len = motion_pred.shape[0]
    for k, v in body_param.items():
        body_param[k] = v.squeeze().cuda()
        body_param[k] = body_param[k][-seq_len:, ...]

    # Get the  prediction from the model
    model_rot_input = (  # (N, 66)
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach()).reshape(motion_pred.shape[0], -1).float()
    )
    assert use_body_part in ["upper", "lower", "full"]
    if use_body_part == "upper":
        pred_full_tmp = torch.zeros((seq_len, 22, 3)).to(model_rot_input.device)
        pred_full_tmp[:, upper_body_part] = model_rot_input.reshape(seq_len, len(upper_body_part), 3)
        model_rot_input = pred_full_tmp.reshape(seq_len, 66)
        body_param['pose_body'].reshape(seq_len, 21, 3)[:, [0, 1, 3, 4, 6, 7, 9, 10]] *= 0.0
    elif use_body_part == "lower":
        pred_full_tmp = torch.zeros((seq_len, 22, 3)).to(model_rot_input.device)
        pred_full_tmp[:, lower_body_part] = model_rot_input.reshape(seq_len, len(lower_body_part), 3)
        model_rot_input = pred_full_tmp.reshape(seq_len, 66)
        body_param['pose_body'].reshape(seq_len, 21, 3)[:, [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]] *= 0.0

    T_head2world = head_motion.clone().cuda()
    t_head2world = T_head2world[:, :3, 3].clone()
    # Get the offset between the head and other joints using forward kinematic model
    # body_pose_local: (batch, 52, 3) joints location
    body_pose_local = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
        }
    ).Jtr

    # Get the offset in global coordiante system between head and body_world.
    t_head2root = -body_pose_local[:, 15, :]  # root - head location
    t_root2world = t_head2root + t_head2world.cuda()

    predicted_body = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            # "trans": body_param['trans'],
            "trans": t_root2world,
        }
    )
    predicted_position = predicted_body.Jtr[:, :22, :]

    # Get the predicted position and rotation
    predicted_angle = model_rot_input

    # Get the  ground truth position from the model
    gt_body = body_model(body_param)
    gt_position = gt_body.Jtr[:, :22, :]

    # Create animation
    # "CMU-94", "CMU-55", "CMU-14", "CMU-206", "MPI_HDM05-20", "BioMotionLab_NTroje-26"
    if args.VIS:
        video_dir = os.path.join(args.SAVE_DIR, "Video")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        save_filename = filename.split(".")[0].replace("/", "-")
        save_video_path = os.path.join(video_dir, save_filename + ".mp4")
        utils_visualize.save_animation(
            body_pose=predicted_body,
            savepath=save_video_path,
            bm=body_model.body_model,
            fps=fps,
            resolution=(800, 800),
        )
        if args.SAVE_GT:
            save_video_path_gt = os.path.join(video_dir, save_filename + "_gt.mp4")
            if not os.path.exists(save_video_path_gt):
                utils_visualize.save_animation(
                    body_pose=gt_body,
                    savepath=save_video_path_gt,
                    bm=body_model.body_model,
                    fps=fps,
                    resolution=(800, 800),

                )
    # import pickle
    # video_dir = "SAGENet/outputs/plot_result"
    # if not os.path.exists(video_dir):
    #     os.makedirs(video_dir)
    #
    # save_filename = filename.split(".")[0].replace("/", "-")
    # save_file_path_gt = os.path.join(video_dir, save_filename + "_SAGENET.pkl")
    # posi = predicted_body.Jtr.cpu().numpy()
    # verts = predicted_body.v.cpu().numpy()
    # faces = predicted_body.f.cpu().numpy()
    # res = {
    #     "pos": posi,
    #     "verts": verts,
    #     "face": faces
    # }
    # file = open(save_file_path_gt, 'wb')
    # print(f"saving {save_file_path_gt}")
    # pickle.dump(res, file)
    # file.close()

    gt_angle = body_param["pose_body"]
    gt_root_angle = body_param["root_orient"]

    predicted_root_angle = predicted_angle[:, :3]
    predicted_angle = predicted_angle[:, 3:]

    eval_log = {}
    for metric in metrics:
        eval_log[metric] = (
            get_metric_function(metric)(
                predicted_position,
                predicted_angle,
                predicted_root_angle,
                gt_position,
                gt_angle,
                gt_root_angle,
                upper_index,
                lower_index,
                fps,
            ).cpu().numpy()
        )

    torch.cuda.empty_cache()
    return eval_log
