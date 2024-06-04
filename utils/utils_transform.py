# MIT License
# Copyright (c) 2022 ETH Sensing, Interaction & Perception Lab
#
# This code is based on https://github.com/eth-siplab/AvatarPoser
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from human_body_prior.tools import tgm_conversion as tgm
from human_body_prior.tools.rotation_tools import aa2matrot, matrot2aa
from torch.nn import functional as F


def bgs(d6s):
    d6s = d6s.reshape(-1, 2, 3).permute(0, 2, 1)
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    c = torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1
    b2 = F.normalize(a2 - c, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrot2sixd(pose_matrot):
    """
    :param pose_matrot: Nx3x3
    :return: pose_6d: Nx6
    """
    pose_6d = torch.cat([pose_matrot[:, :3, 0], pose_matrot[:, :3, 1]], dim=1)
    return pose_6d


def matrot2sixd_single(pose_matrot):
    """
    :param pose_matrot: 3x3
    :return: pose_6d: 6
    """
    pose_6d = torch.cat([pose_matrot[:, 0], pose_matrot[:, 1]], dim=-1)
    return pose_6d


def aa2sixd(pose_aa):
    """
    :param pose_aa Nx3
    :return: pose_6d: Nx6
    """
    pose_matrot = aa2matrot(pose_aa)
    pose_6d = matrot2sixd(pose_matrot)
    return pose_6d


def sixd2matrot(pose_6d):
    """
    :param pose_6d: Nx6
    :return: pose_matrot: Nx3x3
    """
    bs = pose_6d.shape
    rot_vec_1 = pose_6d[:, :3]
    rot_vec_1 = rot_vec_1 / rot_vec_1.norm(dim=-1).reshape(bs[0], 1)
    rot_vec_2 = pose_6d[:, 3:6] - torch.sum(rot_vec_1 * pose_6d[:, 3:6], dim=-1, keepdim=True) * rot_vec_1
    rot_vec_2 = rot_vec_2 / rot_vec_2.norm(dim=-1).reshape(bs[0], 1)
    # rot_vec_2 = pose_6d[:, 3:]
    rot_vec_3 = torch.cross(rot_vec_1, rot_vec_2)  # 叉乘得到第3个基, 共同构成旋转矩阵
    pose_matrot = torch.stack([rot_vec_1, rot_vec_2, rot_vec_3], dim=-1)  # (N, 3, 3)
    return pose_matrot


def sixd2aa(pose_6d, batch=False):
    """
    :param pose_6d: Nx6  (N, 132) -> (N*22, 6) colomn
    :return: pose_aa: Nx3
    """
    if batch:
        B, J, C = pose_6d.shape
        pose_6d = pose_6d.reshape(-1, 6)
    pose_matrot = sixd2matrot(pose_6d)  # (N, 3, 3)
    pose_aa = matrot2aa(pose_matrot)  # (N, 3)
    if batch:
        pose_aa = pose_aa.reshape(B, J, 3)
    return pose_aa


def sixd2quat(pose_6d):
    """
    :param pose_6d: Nx6
    :return: pose_quaternion: Nx4
    """
    pose_mat = sixd2matrot(pose_6d)
    pose_mat_34 = torch.cat(
        (pose_mat, torch.zeros(pose_mat.size(0), pose_mat.size(1), 1)), dim=-1
    )
    pose_quaternion = tgm.rotation_matrix_to_quaternion(pose_mat_34)
    return pose_quaternion


def quat2aa(pose_quat):
    """
    :param pose_quat: Nx4
    :return: pose_aa: Nx3
    """
    return tgm.quaternion_to_angle_axis(pose_quat)


def relSixd2abs(motion):
    # relative pose -> absolute pose  (seq, 132)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    seq = motion.shape[0]
    motion_re = motion.reshape(seq * 22, 6)  # (22, 6)
    motion_matrix = sixd2matrot(motion_re).reshape(seq, 22, 3, 3)  # (seq, 22, 3, 3)
    transform_chain = [motion_matrix[:, 0]]
    for i in range(1, len(parents)):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], motion_matrix[:, i])
        transform_chain.append(curr_res)
    abs_matrix = torch.stack(transform_chain, dim=1)  # (seq, 22, 3, 3)
    abs_matrix = abs_matrix.reshape(-1, 3, 3)
    abs_sixd = matrot2sixd(abs_matrix).reshape(seq, -1)
    return abs_sixd  # (seq, 132)


def absSixd2rel(abs_sixd):
    #  absolute pose ->relative pose  (seq, 132)
    seq = abs_sixd.shape[0]
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    abs_sixd_re = abs_sixd.reshape(seq * 22, 6)  # (seq*22, 6)
    abs_matrix = sixd2matrot(abs_sixd_re).reshape(seq, 22, 3, 3)  # (seq, 22, 3, 3)
    transform_chain = [abs_matrix[:, 0]]
    for i in range(1, len(parents)):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(abs_matrix[:, parents[i]].transpose(-1, -2), abs_matrix[:, i])
        transform_chain.append(curr_res)
    rel_matrix = torch.stack(transform_chain, dim=1).contiguous()  # (seq, 22, 3, 3)
    rel_matrix = rel_matrix.reshape(-1, 3, 3)
    rel_sixd = matrot2sixd(rel_matrix).reshape(seq, -1)
    return rel_sixd  # (seq, 132)
