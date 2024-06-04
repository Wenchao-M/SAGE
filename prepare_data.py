# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os
import numpy as np
import torch

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from tqdm import tqdm
from utils import utils_transform
import glob
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main(args, bm):
    for dataroot_subset in ['ACCAD', 'BioMotionLab_NTroje_train', 'BioMotionLab_NTroje_test', 'BMLmovi', 'CMU_train',
                            'CMU_test', 'EKUT', 'Eyes_Japan_Dataset', 'HumanEva', 'KIT', 'MPI_HDM05_train',
                            'MPI_HDM05_test', 'MPI_Limits', 'MPI_mosh', 'SFU', 'TotalCapture', 'Transitions_mocap']:
        print(dataroot_subset)

        savedir = os.path.join(args.save_dir, dataroot_subset)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if ("train" in dataroot_subset) or ("test" in dataroot_subset):
            split_file = os.path.join("prepare_data/data_split", dataroot_subset + ".txt")
            if os.path.exists(split_file):
                with open(split_file, "r") as f:
                    filepaths = [line.strip() for line in f]
        else:
            filepaths = glob.glob(os.path.join(args.root_dir, dataroot_subset, '**', '*.npz'), recursive=True)

        rotation_local_full_gt_list = []
        hmd_position_global_full_gt_list = []
        body_parms_list = []
        head_global_trans_list = []

        idx = 0
        for filepath in tqdm(filepaths):
            data = {}
            bdata = np.load(
                os.path.join(args.root_dir, filepath), allow_pickle=True
            )

            if "mocap_framerate" in bdata:
                framerate = bdata["mocap_framerate"]
            else:
                continue
            idx += 1

            if framerate == 120:
                stride = 2
            elif framerate == 60:
                stride = 1
            else:
                # raise AssertionError(
                #     "Please check your AMASS data, should only have 2 types of framerate, either 120 or 60!!!"
                # )
                stride = round(framerate / 60)

            bdata_poses = bdata["poses"][::stride, ...]
            bdata_trans = bdata["trans"][::stride, ...]
            subject_gender = bdata["gender"]

            body_parms = {
                "root_orient": torch.Tensor(
                    bdata_poses[:, :3]
                ),  # .to(comp_device), # controls the global root orientation
                "pose_body": torch.Tensor(
                    bdata_poses[:, 3:66]
                ),  # .to(comp_device), # controls the body
                "trans": torch.Tensor(
                    bdata_trans
                ),  # .to(comp_device), # controls the global body position
            }

            body_parms_list = body_parms

            body_pose_world = bm(
                **{
                    k: v.cuda()
                    for k, v in body_parms.items()
                    if k in ["pose_body", "root_orient", "trans"]
                }
            )
            if bdata_poses.shape[0] < 5:
                continue
            output_aa = torch.Tensor(bdata_poses[:, :66]).reshape(-1, 3)
            output_6d = utils_transform.aa2sixd(output_aa).reshape(
                bdata_poses.shape[0], -1
            )
            rotation_local_full_gt_list = output_6d[1:]

            rotation_local_matrot = aa2matrot(
                torch.tensor(bdata_poses).reshape(-1, 3)
            ).reshape(bdata_poses.shape[0], -1, 9)
            rotation_global_matrot = local2global_pose(
                rotation_local_matrot, bm.kintree_table[0].long()
            )  # rotation of joints relative to the origin

            head_rotation_global_matrot = rotation_global_matrot[:, [15], :, :]

            rotation_global_6d = utils_transform.matrot2sixd(
                rotation_global_matrot.reshape(-1, 3, 3)
            ).reshape(rotation_global_matrot.shape[0], -1, 6)
            input_rotation_global_6d = rotation_global_6d[1:, :22, :]  # (seq-1, 3, 6)

            rotation_velocity_global_matrot = torch.matmul(
                torch.inverse(rotation_global_matrot[:-1]),
                rotation_global_matrot[1:],
            )  # (seq-1, 52, 3, 3)
            rotation_velocity_global_6d = utils_transform.matrot2sixd(
                rotation_velocity_global_matrot.reshape(-1, 3, 3)
            ).reshape(rotation_velocity_global_matrot.shape[0], -1, 6)  # (seq-1, 52, 6)
            input_rotation_velocity_global_6d = rotation_velocity_global_6d[:, :22, :]  # (seq-1, 3, 6)

            # position of joints relative to the world origin
            position_global_full_gt_world = body_pose_world.Jtr[:, :22, :].cpu()

            position_head_world = position_global_full_gt_world[:, 15, :]  # world position of head

            head_global_trans = torch.eye(4).repeat(position_head_world.shape[0], 1, 1)  # (seq, 4, 4)
            head_global_trans[:, :3, :3] = head_rotation_global_matrot.squeeze()
            head_global_trans[:, :3, 3] = position_global_full_gt_world[:, 15, :]

            head_global_trans_list = head_global_trans[1:]  # (seq-1, 4, 4)

            num_frames = position_global_full_gt_world.shape[0] - 1  # (seq-1)

            hmd_position_global_full_gt_list = torch.cat(
                [
                    input_rotation_global_6d.reshape(num_frames, -1),  # (seq-1, n*6)
                    input_rotation_velocity_global_6d.reshape(num_frames, -1),  # (seq-1, n*6)
                    position_global_full_gt_world[1:, :22, :].reshape(num_frames, -1),  # (seq-1, n*3)
                    position_global_full_gt_world[1:, :22, :].reshape(num_frames, -1) -
                    position_global_full_gt_world[:-1, :22, :].reshape(num_frames, -1),  # (seq-1, n*3)
                ],
                dim=-1,
            )

            data["rotation_local_full_gt_list"] = rotation_local_full_gt_list
            data["hmd_position_global_full_gt_list"] = hmd_position_global_full_gt_list
            data["body_parms_list"] = body_parms_list
            data["head_global_trans_list"] = head_global_trans_list
            data["position_global_full_gt_world"] = (position_global_full_gt_world[1:].cpu().float())
            data["framerate"] = 60
            data["gender"] = subject_gender
            data["filepath"] = filepath

            torch.save(data, os.path.join(savedir, "{}.pt".format(idx)))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default="body_models",
        help="=dir where you put your smplh and dmpls dirs",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/remote-home/share/fhan/final_test_datadir",
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", type=str, default="/remote-home/share/fhan/AMASS_FULL_ori",
        help="=dir where you put your AMASS data"
    )
    args = parser.parse_args()

    # Here we follow the AvatarPoser paper and use male model for all sequences
    bm_fname_male = os.path.join(args.support_dir, "smplh/{}/model.npz".format("male"))
    dmpl_fname_male = os.path.join(
        args.support_dir, "dmpls/{}/model.npz".format("male")
    )

    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    bm_male = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname_male,
    ).cuda()

    main(args, bm_male)
