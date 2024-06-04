import glob
import os
import random

seed = 0  # here to set random seed
random.seed(seed)


def get_path(dataset_path):
    data_list_path = []
    # parent_data_path = glob.glob(dataset_path + "/*")
    dataset = ['MPI_HDM05_train', 'MPI_HDM05_test', 'BioMotionLab_NTroje_train', 'BioMotionLab_NTroje_test',
               'CMU_train', 'CMU_test', 'ACCAD', 'BMLmovi', 'EKUT', 'Eyes_Japan_Dataset', 'KIT',
               'MPI_Limits', 'MPI_mosh', 'SFU', 'TotalCapture', 'HumanEva', 'Transitions_mocap']

    for d in dataset:
        d = os.path.join(dataset_path, d)
        if os.path.isdir(d):
            files = glob.glob(d + "/" + "/*pt")
            data_list_path.extend(files)
    return data_list_path


def random_split(file_paths):
    # 1.Get full file nums
    total_files = len(file_paths)

    # 2.Calc the number of training samples
    train_size = int(total_files * 0.9)

    # 3.Use random.sample to generate random index
    train_indices = random.sample(range(total_files), train_size)

    # 4.Get train and test set through the index
    train_set = [file_paths[i] for i in train_indices]
    test_set = [file_paths[i] for i in range(total_files) if i not in train_indices]

    # 5.Write results to txt file
    train_file_name = f"prepare_data/data_split/train_full_rand{seed}.txt"
    with open(train_file_name, 'w') as f:
        for path in train_set:
            f.write(path + '\n')
        print(f"Training file save to {train_file_name}.")

    test_file_name = f"prepare_data/data_split/test_full_rand{seed}.txt"
    with open(test_file_name, 'w') as f:
        for path in test_set:
            f.write(path + '\n')
    print(f"Testing file save to {test_file_name}.")
    return train_file_name, test_file_name


def read_from_file(train_file_name, test_fle_name):
    train_set = []
    test_set = []
    with open(train_file_name, 'r') as f:
        for line in f:
            path = line.strip()  # remove line break
            train_set.append(path)

    with open(test_fle_name, 'r') as f:
        for line in f:
            path = line.strip()
            test_set.append(path)

    print(test_set)
    print(os.path.exists(test_set[0]))


if __name__ == '__main__':
    dataset_dir = 'dataset'
    data_file_all = get_path(dataset_dir)
    train_split_name, test_split_name = random_split(data_file_all)
    read_from_file(train_split_name, test_split_name)
