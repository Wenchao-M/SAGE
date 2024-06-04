import copy
import os
from typing import List

import math
import numpy as np
import torch
from utils import utils_transform
import trimesh

origin = (0, 0.0, 0)
# The head, left hand, right hand orientation in Quest2 coordinate system in T-pose
Tpose = torch.tensor([[[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]],

                      [[0., 0., 1.],
                       [1., 0., 0.],
                       [0., 1., 0.]],

                      [[0., 0., -1.],
                       [-1., 0., 0.],
                       [0., 1., 0.]]], dtype=torch.float64)


def load_data(filename):
    fps_all = []
    head_pos_all = []
    head_ori_all = []
    lhand_pos_all = []
    lhand_ori_all = []
    rhand_pos_all = []
    rhand_ori_all = []
    with (open(filename) as f):
        line = f.readline()
        while line:
            tmp = line.strip().split(';')
            if tmp == [""]:  # only a return key
                line = f.readline()
            else:
                fps, head_pos, head_ori, lhand_pos, lhand_ori, rhand_pos, rhand_ori = tmp[0], tmp[1], tmp[2], tmp[3], \
                    tmp[4], tmp[5], tmp[6]
                fps_all.append(fps.split(':')[-1])
                head_pos_all.append(head_pos.split(':')[-1])
                head_ori_all.append(head_ori.split(':')[-1])
                lhand_pos_all.append(lhand_pos.split(':')[-1])
                lhand_ori_all.append(lhand_ori.split(':')[-1])
                rhand_pos_all.append(rhand_pos.split(':')[-1])
                rhand_ori_all.append(rhand_ori.split(':')[-1])
                line = f.readline()
    return fps_all, head_pos_all, head_ori_all, lhand_pos_all, lhand_ori_all, rhand_pos_all, rhand_ori_all


def pos2np(data):
    assert isinstance(data, List)
    first_all = []
    second_all = []
    third_all = []
    for head_it in data:
        tmp = head_it.split('(')[-1][:-1].split(',')
        x = float(tmp[0]) - origin[0]
        y = float(tmp[1]) - origin[1]
        z = float(tmp[2]) - origin[2]
        first_all.append(x)
        second_all.append(y)
        third_all.append(z)
    x_all_arr = np.array(first_all)
    y_all_arr = np.array(second_all)
    z_all_arr = np.array(third_all)

    pos_np = np.stack((-x_all_arr, y_all_arr, z_all_arr)).T  # trans left hand coord to right hand coord
    return pos_np


def angle2np(data):
    assert isinstance(data, List)
    x_all = []
    y_all = []
    z_all = []
    w_all = []
    for head_it in data:
        tmp = head_it.split('(')[-1][:-1].split(',')
        x = float(tmp[0])
        y = float(tmp[1])
        z = float(tmp[2])
        w = float(tmp[3])
        x_all.append(x)
        y_all.append(y)
        z_all.append(z)
        w_all.append(w)
    x_all_arr = np.array(x_all)
    y_all_arr = np.array(y_all)
    z_all_arr = np.array(z_all)
    w_all_arr = np.array(w_all)

    # unity use left-handed coordinate systems, xyz correspond to (z, -x, y)
    # pyrender use right-handed coordinate systems, xyz correspond to (z, x, y)
    angle_np = np.stack((w_all_arr, x_all_arr, -y_all_arr, -z_all_arr)).T  # trans left hand coord to right hand coord
    return angle_np


def mat2homo(mat, trans):
    # mat: (seq, 3, 3)  trans:(seq, 3)
    seq = mat.shape[0]
    init_mat = torch.cat((mat, trans[..., None]), dim=-1)  # (seq, 3, 4)
    last_row = torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(seq, 1, 1)  # (seq, 1, 4)
    res = torch.cat((init_mat, last_row), dim=-2)
    return res


def quaternion_to_rotation_matrix(q):
    r"""
    Turn (unnormalized) quaternions wxyz into rotation matrices. (torch, batch)

    :param q: Quaternion tensor that can reshape to [batch_size, 4].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    r = torch.cat((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                   2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
                   2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), dim=1)
    return r.view(-1, 3, 3)


file_dir = "/remote-home/fenghan/cvpr2024/SAGENet/livedemo/data0325"
file_names = os.listdir(file_dir)
for file_name in file_names:
    if file_name.endswith(".pt"):
        continue
    file_name = os.path.join(file_dir, file_name)
    fps_list, head_pos_list, head_ori_list, lhand_pos_list, lhand_ori_list, rhand_pos_list, rhand_ori_list = load_data(
        file_name)
    head_pos_np = torch.from_numpy(pos2np(head_pos_list))
    lhand_pos_np = torch.from_numpy(pos2np(lhand_pos_list))
    rhand_pos_np = torch.from_numpy(pos2np(rhand_pos_list))
    head_ori_np = torch.from_numpy(angle2np(head_ori_list))
    lhand_ori_np = torch.from_numpy(angle2np(lhand_ori_list))
    rhand_ori_np = torch.from_numpy(angle2np(rhand_ori_list))

    head_raw_matrix = quaternion_to_rotation_matrix(head_ori_np)
    lhand_raw_matrix = quaternion_to_rotation_matrix(lhand_ori_np)
    rhand_raw_matrix = quaternion_to_rotation_matrix(rhand_ori_np)
    full_raw_matrix = torch.cat([head_raw_matrix[:, None], lhand_raw_matrix[:, None], rhand_raw_matrix[:, None]],
                                dim=1)  # (seq, 3, 3, 3)
    full_cal_matrix = full_raw_matrix.matmul(Tpose.transpose(-1, -2)[None])  # coordinate transform
    full_cal_matrix_homo = torch.eye(4)[None, None].repeat(full_cal_matrix.shape[0], 3, 1, 1).to(float)
    full_cal_matrix_homo[..., :3, :3] = full_cal_matrix
    full_cal_matrix_homo[..., :3, 3:] = torch.cat(  # (seq, 3, 3, 1)
        [head_pos_np[:, None, :, None], lhand_pos_np[:, None, :, None], rhand_pos_np[:, None, :, None]], dim=1)
    # align with training dataset
    transform_all = torch.from_numpy(trimesh.transformations.rotation_matrix(np.pi / 2, (10, 0, 0)))
    transform_all = transform_all[None, None]
    full_final_matrix_homo = torch.matmul(transform_all, full_cal_matrix_homo)

    head_ori_mat = full_final_matrix_homo[:, 0, :3, :3]  # (seq, 3, 3)
    lhand_ori_mat = full_final_matrix_homo[:, 1, :3, :3]
    rhand_ori_mat = full_final_matrix_homo[:, 2, :3, :3]
    head_pos_np = full_final_matrix_homo[:, 0, :3, -1]  # (seq, 3)
    lhand_pos_np = full_final_matrix_homo[:, 1, :3, -1]
    rhand_pos_np = full_final_matrix_homo[:, 2, :3, -1]

    # 6d representation
    head_ori_6d = utils_transform.matrot2sixd(head_ori_mat)
    lhand_ori_6d = utils_transform.matrot2sixd(lhand_ori_mat)
    rhand_ori_6d = utils_transform.matrot2sixd(rhand_ori_mat)

    # calculate velocity
    head_pos_vel = head_pos_np[1:] - head_pos_np[:-1]
    lhand_pos_vel = lhand_pos_np[1:] - lhand_pos_np[:-1]
    rhand_pos_vel = rhand_pos_np[1:] - rhand_pos_np[:-1]

    head_ori_vel = torch.matmul(torch.inverse(head_ori_mat[:-1]), head_ori_mat[1:])
    lhand_ori_vel = torch.matmul(torch.inverse(lhand_ori_mat[:-1]), lhand_ori_mat[1:])
    rhand_ori_vel = torch.matmul(torch.inverse(rhand_ori_mat[:-1]), rhand_ori_mat[1:])
    head_ori_vel_6d = utils_transform.matrot2sixd(head_ori_vel)
    lhand_ori_vel_6d = utils_transform.matrot2sixd(lhand_ori_vel)
    rhand_ori_vel_6d = utils_transform.matrot2sixd(rhand_ori_vel)

    head_all = torch.cat((head_ori_6d[1:], head_ori_vel_6d, head_pos_np[1:], head_pos_vel), dim=-1)
    lhand_all = torch.cat((lhand_ori_6d[1:], lhand_ori_vel_6d, lhand_pos_np[1:], lhand_pos_vel), dim=-1)
    rhand_all = torch.cat((rhand_ori_6d[1:], rhand_ori_vel_6d, rhand_pos_np[1:], rhand_pos_vel), dim=-1)
    sparse_all = torch.cat((head_all, lhand_all, rhand_all), dim=-1)

    print(sparse_all.shape)
    torch.save(sparse_all[100:], file_name + "_new0505.pt")
    print(f"saving {file_name}_new0505.pt")
