import glob
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self, motions, sparses, input_motion_length=196, full_motion_len=1, normalization=True):
        self.motions = motions
        self.sparses = sparses
        self.full_motion_len = full_motion_len
        self.normalization = normalization
        self.input_motion_length = input_motion_length

        self.up_idx = [15, 20, 21]
        self.motion_54 = []
        self.n_joint = 22
        for idx in tqdm(range(len(self.sparses))):
            sparse = self.sparses[idx]
            # rot_rel = self.motions[idx].reshape(-1, self.n_joint, 6)
            rot_abs = sparse[:, :self.n_joint * 6].reshape(-1, self.n_joint, 6)
            rot_vel = sparse[:, self.n_joint * 6: self.n_joint * 12].reshape(-1, self.n_joint, 6)
            pos = sparse[:, self.n_joint * 12: self.n_joint * 15].reshape(-1, self.n_joint, 3)
            pos_vel = sparse[:, self.n_joint * 15: self.n_joint * 18].reshape(-1, self.n_joint, 3)
            # sparse_396 = torch.cat((rot_rel, rot_vel, pos, pos_vel), dim=-1)  # (seq, 22, 18)
            sparse_396_abs = torch.cat((rot_abs, rot_vel, pos, pos_vel), dim=-1)
            # self.motion_396.append(sparse_396)  # (seq, 22, 18)
            self.motion_54.append(sparse_396_abs[:, self.up_idx])

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion_132 = self.motions[idx].float().reshape(-1, 132)
        sparse = self.motion_54[idx].float()
        seqlen = motion_132.shape[0]

        if seqlen <= self.full_motion_len:
            i = 0
        else:
            i = torch.randint(0, int(seqlen - self.full_motion_len), (1,))[0]

        motion_132_all = motion_132[i: i + self.full_motion_len]  # (seq, 22, 18)
        sparse_all = sparse[i: i + self.full_motion_len]
        rel_len = motion_132_all.shape[0]
        num = rel_len - self.input_motion_length + 1

        motion_132_input = []
        sparse_input = []
        for j in range(num):
            motion_132_input.append(motion_132_all[j:j + self.input_motion_length])
            sparse_input.append(sparse_all[j:j + self.input_motion_length])

        motion_132_input = torch.stack(motion_132_input)  # 正常为(500, 20, 396)
        sparse_input = torch.stack(sparse_input)

        # motion_396: (500, 20, 132) sparse:(500, 20, 3, 18)
        return motion_132_input, sparse_input


class TestDataset(Dataset):
    def __init__(self, all_info, filename_list):
        self.filename_list = filename_list
        self.motions = []
        self.sparses = []
        self.body_params = []
        self.head_motion = []
        for i in all_info:
            self.motions.append(i["rotation_local_full_gt_list"])
            self.sparses.append(i["hmd_position_global_full_gt_list"])
            self.body_params.append(i["body_parms_list"])
            self.head_motion.append(i["head_global_trans_list"])

        self.motion_54 = []
        self.up_idx = [15, 20, 21]
        self.n_joint = 22
        for idx in range(len(self.sparses)):
            sparse = self.sparses[idx]
            seq = sparse.shape[0]
            # rot_absolute = self.motions[idx].reshape(-1, self.n_joint, 6)
            rot_abs = sparse[:, :self.n_joint * 6].reshape(-1, self.n_joint, 6)
            rot_vel = sparse[:, self.n_joint * 6: self.n_joint * 12].reshape(-1, self.n_joint, 6)
            pos = sparse[:, self.n_joint * 12: self.n_joint * 15].reshape(-1, self.n_joint, 3)
            pos_vel = sparse[:, self.n_joint * 15: self.n_joint * 18].reshape(-1, self.n_joint, 3)
            # sparse_396 = torch.cat((rot_absolute, rot_vel, pos, pos_vel), dim=-1).reshape(seq, -1)  # (seq, 22, 18)
            sparse_396_abs = torch.cat((rot_abs, rot_vel, pos, pos_vel), dim=-1)
            self.motion_54.append(sparse_396_abs[:, self.up_idx])

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion = self.motions[idx]  # (N, 132)
        # motion_abs = self.motion_absolute[idx]
        # motion_abs = self.sparses[idx][:, :132]
        sparse = self.motion_54[idx]  # (N, 54)
        body_param = self.body_params[idx]  # {root_orient:(N+1,3), pose_body:(N+1,63), trans:(N+1,3)}
        head_motion = self.head_motion[idx]  # (N, 4, 4)
        filename = self.filename_list[idx]

        return motion, sparse, body_param, head_motion, filename


def get_motion(motion_list):
    motions = [i["rotation_local_full_gt_list"] for i in motion_list]  # list((N,132))
    sparses = [i["hmd_position_global_full_gt_list"] for i in motion_list]  # list((N,54))
    return motions, sparses


def get_path(dataset_path, split, protocol):
    data_list_path = []
    assert split in ["train", "test"]
    assert protocol in ['1_small_dataset', '2_full_dataset', 'real', 'randomsplit_0']
    if protocol == '1_small_dataset':
        if split == "train":
            dataset = ['CMU_train', 'BioMotionLab_NTroje_train', 'MPI_HDM05_train']
        elif split == "test":
            dataset = ['CMU_test', 'BioMotionLab_NTroje_test', 'MPI_HDM05_test']
    elif protocol == '2_full_dataset':
        if split == "train":
            dataset = ['MPI_HDM05_train', 'MPI_HDM05_test', 'BioMotionLab_NTroje_train', 'BioMotionLab_NTroje_test',
                       'CMU_train', 'CMU_test', 'ACCAD', 'BMLmovi', 'EKUT', 'Eyes_Japan_Dataset', 'KIT',
                       'MPI_Limits', 'MPI_mosh', 'SFU', 'TotalCapture']
        elif split == "test":
            dataset = ['HumanEva', 'Transitions_mocap']
    elif protocol == 'real':
        dataset = ['real']
    elif protocol == 'randomsplit_0':
        if split == 'train':
            dataset_file_name = 'prepare_data/data_split/train_full_rand0.txt'
        else:
            dataset_file_name = 'prepare_data/data_split/test_full_rand0.txt'
        print(f"{split} using {dataset_file_name}")
        with open(dataset_file_name, 'r') as f:
            for line in f:
                path = line.strip()
                data_list_path.append(path)
        return data_list_path
    else:
        return

    print(f"{split} using {dataset}")
    for d in dataset:
        d = os.path.join(dataset_path, d)
        if os.path.isdir(d):
            files = glob.glob(d + "/" + "/*pt")
            data_list_path.extend(files)
    return data_list_path


def load_data(dataset_path, split, protocol, **kwargs):
    if split == "test":
        motion_list = get_path(dataset_path, split, protocol)
        filename_list = [
            "-".join([i.split("/")[-3], i.split("/")[-1]]).split(".")[0]
            for i in motion_list
        ]
        motion_list = [torch.load(i) for i in tqdm(motion_list)]
        return filename_list, motion_list

    assert split == "train"
    assert ("input_motion_length" in kwargs), "Please specify the input_motion_length"

    motion_list = get_path(dataset_path, split, protocol)
    input_motion_length = kwargs["input_motion_length"]
    motion_list = [torch.load(i) for i in tqdm(motion_list)]

    motions, sparses = get_motion(motion_list)

    new_motions = []
    new_sparses = []
    for idx, motion in enumerate(motions):
        if motion.shape[0] < input_motion_length:  # Arbitrary choice
            continue
        new_sparses.append(sparses[idx])
        new_motions.append(motions[idx])

    return new_motions, new_sparses, motion_list


def get_dataloader(dataset, split, batch_size, num_workers=32):
    if split == "train":
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader
