# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Metric functions with same inputs

import numpy as np
import torch
from utils import utils_transform


def pred_jitter(predicted_position, predicted_angle, predicted_root_angle, gt_position,
                gt_angle, gt_root_angle, upper_index, lower_index, fps, ):
    pred_jitter = (
        ((predicted_position[3:] - 3 * predicted_position[2:-1] +
          3 * predicted_position[1:-2] - predicted_position[:-3]) * (fps ** 3)).norm(dim=2).mean()
    )
    return pred_jitter


def gt_jitter(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
              gt_root_angle, upper_index, lower_index, fps, ):
    gt_jitter = (
        (
                (gt_position[3:] - 3 * gt_position[2:-1] +
                 3 * gt_position[1:-2] - gt_position[:-3]) * (fps ** 3)).norm(dim=2).mean()
    )
    return gt_jitter


def mpjre(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
          gt_root_angle, upper_index, lower_index, fps, ):
    diff = gt_angle - predicted_angle
    diff = utils_transform.matrot2aa(utils_transform.aa2matrot(diff.reshape(-1, 3)))
    # diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    # diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


def rootre(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
           gt_root_angle, upper_index, lower_index, fps, ):
    diff = gt_root_angle - predicted_root_angle
    diff = utils_transform.matrot2aa(utils_transform.aa2matrot(diff))
    # diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    # diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


def mpjpe(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
          gt_root_angle, upper_index, lower_index, fps, ):
    pos_error = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1)))
    return pos_error


def handpe(predicted_position, predicted_angle, predicted_root_angle, gt_position,
           gt_angle, gt_root_angle, upper_index, lower_index, fps, ):
    pos_error_hands = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[..., [20, 21]]
    )
    return pos_error_hands


def upperpe(predicted_position, predicted_angle, predicted_root_angle, gt_position,
            gt_angle, gt_root_angle, upper_index, lower_index, fps, ):
    upper_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[..., upper_index]
    )
    return upper_body_error


def lowerpe(predicted_position, predicted_angle, predicted_root_angle, gt_position,
            gt_angle, gt_root_angle, upper_index, lower_index, fps, ):
    lower_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[..., lower_index]
    )
    return lower_body_error


def rootpe(predicted_position, predicted_angle, predicted_root_angle, gt_position,
           gt_angle, gt_root_angle, upper_index, lower_index, fps, ):
    pos_error_root = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[..., [0]]
    )
    return pos_error_root


def mpjve(predicted_position, predicted_angle, predicted_root_angle, gt_position,
          gt_angle, gt_root_angle, upper_index, lower_index, fps, ):
    gt_velocity = (gt_position[1:, ...] - gt_position[:-1, ...]) * fps
    predicted_velocity = (predicted_position[1:, ...] - predicted_position[:-1, ...]) * fps
    vel_error = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity - predicted_velocity), axis=-1)))
    return vel_error


def upperre(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
            gt_root_angle, upper_index, lower_index, fps, ):
    if predicted_angle.shape[-1] == 63:
        gt_angle_tmp = torch.cat((gt_root_angle, gt_angle), dim=-1)
        pred_angle_tmp = torch.cat((predicted_root_angle, predicted_angle), dim=-1)
    else:
        gt_angle_tmp = gt_angle
        pred_angle_tmp = predicted_angle
    diff = (gt_angle_tmp - pred_angle_tmp).reshape(-1, 22, 3)[:, upper_index]
    diff = utils_transform.matrot2aa(utils_transform.aa2matrot(diff.reshape(-1, 3)))
    # diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    # diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


def lowerre(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
            gt_root_angle, upper_index, lower_index, fps, ):
    if predicted_angle.shape[-1] == 63:
        gt_angle_tmp = torch.cat((gt_root_angle, gt_angle), dim=-1)
        pred_angle_tmp = torch.cat((predicted_root_angle, predicted_angle), dim=-1)
    else:
        gt_angle_tmp = gt_angle
        pred_angle_tmp = predicted_angle
    diff = (gt_angle_tmp - pred_angle_tmp).reshape(-1, 22, 3)[:, lower_index]
    diff = utils_transform.matrot2aa(utils_transform.aa2matrot(diff.reshape(-1, 3)))
    # diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    # diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


metric_funcs_dict = {
    "mpjre": mpjre,
    "rootre": rootre,
    "upperre": upperre,
    "lowerre": lowerre,
    "mpjpe": mpjpe,
    "mpjve": mpjve,
    "handpe": handpe,
    "upperpe": upperpe,
    "lowerpe": lowerpe,
    "rootpe": rootpe,
    "pred_jitter": pred_jitter,
    "gt_jitter": gt_jitter,
}


def get_metric_function(metric):
    return metric_funcs_dict[metric]
