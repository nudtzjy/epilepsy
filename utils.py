'''
author: zhangjunyang
time: 20241119
'''
import pickle
import math
import os
import random
import numpy as np
import argparse
from logger import logger
from scipy import io
import glob
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

scaler_std = StandardScaler()


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/dataConfig.yaml")
    FLAGS = parser.parse_args()
    return FLAGS


def read_data_content(file_path):
    mat_data = io.loadmat(file_path)
    mat_data_val = mat_data['data_convert']
    data = np.array(mat_data_val)
    return data


def get_file_path_list_glob(eeg_data_folder):
    file_path_list = glob.glob(os.path.join(eeg_data_folder, '*'))
    file_path_list_all = []
    for file_path in file_path_list:
        file_list = os.listdir(file_path)
        for file_name in file_list:
            file_path_list_all.append(os.path.join(file_path, file_name))
    return file_path_list_all


def get_file_path_list(eeg_data_folder):
    folder_list = os.listdir(eeg_data_folder)
    file_path_list = []
    for folder in folder_list:
        folder_path = os.path.join(eeg_data_folder, folder)
        file_list = os.listdir(folder_path)
        for file in file_list:
            file_path = os.path.join(folder_path, file)
            if '.mat' not in file:
                continue
            file_path_list.append([folder, file_path])
    return file_path_list


def get_data(config, eeg_data_folder, random_seed, split_portion, data_size=-1, use_cache=True, issplit=True):
    feature_cache_path = str(config['dataPath']['feature_cache_path'])
    split_ssl_path = str(config['dataPath']['split_ssl_path'])
    os.makedirs(split_ssl_path, exist_ok=True)
    if use_cache and os.path.exists(feature_cache_path):
        with open(feature_cache_path, 'rb') as f:
            data_list = pickle.load(f)
    else:
        file_path_list = get_file_path_list(eeg_data_folder)
        data_list = []
        count = 0
        for item in file_path_list:
            if count % 1000 == 0:
                logger.info('count: {}'.format(count))
            folder, file_path = item
            sample_data = read_data_content(file_path)
            data_list.append([folder, file_path, sample_data])
            count += 1
        length_count_dict, length_count_list = {}, []
        for data in data_list:
            length = len(data[2])
            if length not in length_count_dict:
                length_count_dict[length] = 0
            length_count_dict[length] += 1
            length_count_list.append(length)
        logger.info(
            'avg length: {}, median length: {}'.format(np.average(length_count_list), np.median(length_count_list)))
        logger.info('data_size: {}'.format(len(data_list)))
        logger.info(length_count_dict)

        with open(feature_cache_path, 'wb') as f:
            pickle.dump(data_list, f)

    logger.info('start to split data to train,valid,test ')
    if issplit:
        if random_seed is not None:
            idx_list = [i for i in range(len(data_list))]
            random.Random(random_seed).shuffle(idx_list)
            new_data_list = []
            for idx in idx_list:
                new_data_list.append(data_list[idx])
                if 0 < data_size < len(new_data_list):
                    break
            data_list = new_data_list

        train_ratio, valid_ratio, test_ratio = split_portion
        valid_start_idx = math.ceil(len(data_list) * train_ratio)
        test_start_idx = math.ceil(len(data_list) * (train_ratio + valid_ratio))
        ssl_train_data = data_list[: valid_start_idx]
        ssl_valid_data = data_list[valid_start_idx: test_start_idx]
        ssl_test_data = data_list[test_start_idx:]

        logger.info('start to save split ssl data to train,valid,test ')
        with open(split_ssl_path + 'ssl_train.pkl', 'wb') as f:
            pickle.dump(ssl_train_data, f)
        with open(split_ssl_path + 'ssl_valid.pkl', 'wb') as f:
            pickle.dump(ssl_valid_data, f)
        with open(split_ssl_path + 'ssl_test.pkl', 'wb') as f:
            pickle.dump(ssl_test_data, f)

        logger.info('save split ssl data to train,valid,test  over ')


def softmax(x):
    exp_values = np.exp(x)
    softmax_values = exp_values / np.sum(exp_values)
    return softmax_values


class ModelDataset(Dataset):
    def __init__(self, data, is_train=True, is_test=False):
        self.data = data
        self.is_test = is_test
        self.is_train = is_train
        data_train, labels_train = self.data['train_feature'], self.data['train_label']
        data_test, labels_test = self.data['test_feature'], self.data['test_label']
        data_val, labels_val = self.data['valid_feature'], self.data['valid_label']
        if is_test is True:
            print("Testing data:")
            self.imgs, self.lbls = data_test, labels_test
        elif is_train is True:
            print("Training data:")
            self.imgs, self.lbls = data_train, labels_train
        else:
            print("Val data:")
            self.imgs, self.lbls = data_val, labels_val

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data = self.imgs[idx]
        label = self.lbls[idx]
        return [data, label]


def read_data_content(file_path):
    mat_data = io.loadmat(file_path)
    mat_data_val = mat_data['data_convert']
    data = np.array(mat_data_val)
    return data


def read_data_content_data_convert(file_path):
    mat_data = io.loadmat(file_path)
    mat_data_val = mat_data['data_convert']
    data = np.array(mat_data_val)
    return data


def continues_select_std_matrix(matrix, window_size, stride):
    """
    根据输入的二维NumPy数组、截取长度和步长，截取矩阵的子矩阵。
    参数:
    - matrix: 二维NumPy数组，输入的矩阵。
    - window_size: 整数，每次截取的子矩阵的列数。
    - stride: 整数，两次截取之间的步长（以列数为单位）。
    返回:
    - 一个列表，包含截取的子矩阵,after std。
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("输入的matrix必须是一个二维NumPy数组")
    if matrix.shape[1] < window_size:
        raise ValueError("矩阵的列数必须大于或等于window_size")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size和stride必须大于0")
    samples = []
    global_start_index = 0
    while global_start_index < matrix.shape[1] - window_size + 1:
        sample = matrix[:, global_start_index:global_start_index + window_size]
        sample = scaler_std.fit_transform(sample)
        samples.append(sample)
        # update global index
        global_start_index += stride
    return samples


def continues_select_nostd_matrix(matrix, window_size, stride):
    """
    根据输入的二维NumPy数组、截取长度和步长，截取矩阵的子矩阵。
    参数:
    - matrix: 二维NumPy数组，输入的矩阵。
    - window_size: 整数，每次截取的子矩阵的列数。
    - stride: 整数，两次截取之间的步长（以列数为单位）。
    返回:
    - 一个列表，包含截取的子矩阵,after std。
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("输入的matrix必须是一个二维NumPy数组")
    if matrix.shape[1] < window_size:
        raise ValueError("矩阵的列数必须大于或等于window_size")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size和stride必须大于0")
    samples = []
    global_start_index = 0
    while global_start_index < matrix.shape[1] - window_size + 1:
        sample = matrix[:, global_start_index:global_start_index + window_size]
        samples.append(sample)
        # update global index
        global_start_index += stride
    return samples


def continues_select_nostd_line(line, window_size, stride):
    samples = []
    global_start_index = 0
    while global_start_index < line.shape[1] - window_size + 1:
        sample = line[:, global_start_index:global_start_index + window_size]
        samples.append(sample)
        # update global index
        global_start_index += stride

    return samples


def sample_matrix_with_global_indices_std(matrix, window_size, stride):
    """
    根据输入的二维NumPy数组、截取长度和步长，截取矩阵的子矩阵，并输出每个采样的全局起始索引。
    参数:
    - matrix: 二维NumPy数组，输入的矩阵。
    - window_size: 整数，每次截取的子矩阵的列数。
    - stride: 整数，两次截取之间的步长（以列数为单位）。
    返回:
    - 一个列表的元组，每个元组包含截取的子矩阵和对应的全局起始索引。
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("输入的matrix必须是一个二维NumPy数组")
    if matrix.shape[1] < window_size:
        raise ValueError("矩阵的列数必须大于或等于window_size")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size和stride必须大于0")
    samples_with_indices = []
    global_start_index = 0
    while global_start_index < matrix.shape[1] - window_size + 1:
        sample = matrix[:, global_start_index:global_start_index + window_size]
        sample = scaler_std.fit_transform(sample)
        samples_with_indices.append((sample, global_start_index))
        global_start_index += stride
    return samples_with_indices


def sample_matrix_with_global_indices_nostd(matrix, window_size, stride):
    """
    根据输入的二维NumPy数组、截取长度和步长，截取矩阵的子矩阵，并输出每个采样的全局起始索引。
    参数:
    - matrix: 二维NumPy数组，输入的矩阵。
    - window_size: 整数，每次截取的子矩阵的列数。
    - stride: 整数，两次截取之间的步长（以列数为单位）。
    返回:
    - 一个列表的元组，每个元组包含截取的子矩阵和对应的全局起始索引。
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("输入的matrix必须是一个二维NumPy数组")
    if matrix.shape[1] < window_size:
        raise ValueError("矩阵的列数必须大于或等于window_size")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size和stride必须大于0")

    samples_with_indices = []
    global_start_index = 0
    while global_start_index < matrix.shape[1] - window_size + 1:
        sample = matrix[:, global_start_index:global_start_index + window_size]
        samples_with_indices.append((sample, global_start_index))
        global_start_index += stride

    return samples_with_indices


def sample_line_with_global_indices_nostd(line, window_size, stride):
    """
    根据输入的一维NumPy数组、截取长度和步长，截取矩阵的子矩阵，并输出每个采样的全局起始索引。
    参数:
    - line: 一维NumPy数组，输入的矩阵。
    - window_size: 整数，每次截取的子矩阵的列数。
    - stride: 整数，两次截取之间的步长（以列数为单位）。
    返回:
    - 一个列表的元组，每个元组包含截取的子矩阵和对应的全局起始索引。
    """
    if line.shape[1] < window_size:
        raise ValueError("列数必须大于或等于window_size")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size和stride必须大于0")
    samples_with_indices = []
    global_start_index = 0
    while global_start_index < line.shape[1] - window_size + 1:
        sample = line[:, global_start_index:global_start_index + window_size]
        samples_with_indices.append((sample, global_start_index))
        global_start_index += stride
    return samples_with_indices


def sample_line_with_global_indices_std(line, window_size, stride):
    """
    根据输入的yi维NumPy数组、截取长度和步长，截取矩阵的子矩阵，并输出每个采样的全局起始索引。
    参数:
    - line: yi维NumPy数组，输入的矩阵。
    - window_size: 整数，每次截取的子矩阵的列数。
    - stride: 整数，两次截取之间的步长（以列数为单位）。
    返回:
    - 一个列表的元组，每个元组包含截取的子矩阵和对应的全局起始索引。
    """
    if line.shape[1] < window_size:
        raise ValueError("列数必须大于或等于window_size")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size和stride必须大于0")
    samples_with_indices = []
    global_start_index = 0
    while global_start_index < line.shape[1] - window_size + 1:
        sample = line[:, global_start_index:global_start_index + window_size]
        sample = scaler_std.fit_transform(sample)
        samples_with_indices.append((sample, global_start_index))
        global_start_index += stride
    return samples_with_indices


def random_select(timeser, need_lenth):
    rnd = np.random.random()
    time_len = timeser.shape[1]
    mask_index = np.array(random.sample(list(np.arange(0, time_len)), need_lenth))
    bool_mask = np.zeros((time_len))
    bool_mask[mask_index] = 1
    bool_mask = bool_mask.astype(bool)

    return timeser[:, bool_mask]


def get_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths
