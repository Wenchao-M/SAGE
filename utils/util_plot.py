import torch
import os
import matplotlib.pyplot as plt
from utils import utils_transform
from human_body_prior.body_model.body_model import BodyModel as BM

line_between_point = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11],
                      [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20],
                      [19, 21]]


class BodyModel(torch.nn.Module):
    def __init__(self, support_dir):
        super().__init__()

        device = torch.device("cuda")
        subject_gender = "male"
        bm_fname = os.path.join(
            support_dir, "smplh/{}/model.npz".format(subject_gender)
        )
        dmpl_fname = os.path.join(
            support_dir, "dmpls/{}/model.npz".format(subject_gender)
        )
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        body_model = BM(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        ).to(device)
        self.body_model = body_model.eval()

    def forward(self, body_params):  # body_params:{pose_body:(N, 63), root_orient:(N, 3)}
        with torch.no_grad():
            body_pose = self.body_model(
                **{
                    k: v
                    for k, v in body_params.items()
                    if k in ["pose_body", "trans", "root_orient", "joints"]
                }
            )
        # body_pose:由字典包装成为类
        # body_pose.v: (batch, 6890, 3)  vertice的位置
        # body_pose.f: (13776, 3)
        # body_pose.Jtr: (batch, 52, 3) joints的位置
        # body_pose.full_pose: (batch, 156) 预测出来的22个身体pose + 30个全0的 hand pose
        return body_pose


def plot_skeleton(pred_motion, gt_motion):
    if len(pred_motion.shape) > 2:
        pred_motion = pred_motion[0]  # (seq, 132)
        gt_motion = gt_motion[0]
    seq_len = pred_motion.shape[0]
    support_dir = '/home/fhan/repo/amass/body_models'
    body_model = BodyModel(support_dir)
    pred_motion_3 = utils_transform.sixd2aa(pred_motion.reshape(seq_len, 22, 6), batch=True).reshape(-1, 66)
    gt_motion_3 = utils_transform.sixd2aa(gt_motion.reshape(seq_len, 22, 6), batch=True).reshape(-1, 66)
    pred_jtr = body_model({
        "root_orient": pred_motion_3[..., :3],
        "pose_body": pred_motion_3[..., 3:],
    }).Jtr[:, :22].detach().cpu()
    gt_jtr = body_model({
        "root_orient": gt_motion_3[..., :3],
        "pose_body": gt_motion_3[..., 3:],
    }).Jtr[:, :22].detach().cpu()

    p1_all = pred_jtr
    p2_all = gt_jtr
    for i in range(10):
        p1 = p1_all[i]
        p2 = p2_all[i]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2])
        ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2])
        for line in line_between_point:
            i = line[0]
            j = line[1]
            ax.plot([p1[i, 0], p1[j, 0]], [p1[i, 1], p1[j, 1]], [p1[i, 2], p1[j, 2]], 'r-')
            ax.plot([p2[i, 0], p2[j, 0]], [p2[i, 1], p2[j, 1]], [p2[i, 2], p2[j, 2]], 'b-')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 显示图形
        plt.show()


def plot_skeleton_by_pos(pred_jtr, gt_jtr):
    # (seq, 22, 3)
    if len(pred_jtr.shape) > 3:  # (bs, seq, 22, 3)
        pred_jtr = pred_jtr[0].cpu()
        gt_jtr = gt_jtr[0].cpu()
    p1_all = pred_jtr.cpu()
    p2_all = gt_jtr.cpu()
    for i in range(10):
        p1 = p1_all[i]
        p2 = p2_all[i]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2])
        ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2])
        for line in line_between_point:
            i = line[0]
            j = line[1]
            ax.plot([p1[i, 0], p1[j, 0]], [p1[i, 1], p1[j, 1]], [p1[i, 2], p1[j, 2]], 'r-')
            ax.plot([p2[i, 0], p2[j, 0]], [p2[i, 1], p2[j, 1]], [p2[i, 2], p2[j, 2]], 'b-')
        # generate labels for axises
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # show
        plt.show()
