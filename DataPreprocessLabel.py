"""
time: 20241125
aim to get training dataset to (data, label) for epilepsy model
"""
import pickle
import math
import os
import random
import argparse
import yaml
from logger import logger
from utils import get_file_path_list_glob, read_data_content, continues_select_std_matrix, \
    continues_select_nostd_matrix, random_select


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/dataConfig.yaml")
    FLAGS = parser.parse_args()
    return FLAGS


def get_feature_data(config):
    feature_folder = str(config['dataPath']['feature_folder'])
    processed_path = str(config['dataPath']['processed_path'])
    os.makedirs(processed_path, exist_ok=True)
    # read data
    file_list = get_file_path_list_glob(feature_folder)
    feature0_to_write, label0_to_write = [], []
    feature1_to_write, label1_to_write = [], []
    for file_tuple in file_list:
        lab = file_tuple.strip().split('/')[-2]
        label = int(lab)
        feature = read_data_content(file_tuple)
        if label == 0:
            feature0_to_write.append(feature)
        elif label == 1:
            feature1_to_write.append(feature)

    # data enhancement
    data0_aug_list = []
    data1_aug_list = []
    need_length = config['dataConfig']['need_length']
    stride_0 = config['dataConfig']['stride_0']
    stride_1 = config['dataConfig']['stride_1']

    # baseline data process
    count = 0
    for line in feature0_to_write:
        if line.shape[1] < need_length:
            continue
        select_data = continues_select_nostd_matrix(line, need_length, stride_0)
        data0_aug_list.extend(select_data)
        if count % 5 == 0:
            logger.info('0 sample count is : {}'.format(count))
        count += 1

    label0_aug_list = [0] * len(data0_aug_list)

    count = 0
    # This value was reasonably set according to the number of seizure and non seizure samples
    num_1_sample = 20000
    for line in feature1_to_write:
        if line.shape[1] < need_length:
            continue
        select_data = continues_select_nostd_matrix(line, need_length, stride_1)
        data1_aug_list.extend(select_data)
        for j in range(num_1_sample):
            each_select = random_select(line, 100)
            data1_aug_list.append(each_select)
        if count % 5 == 0:
            logger.info('1 sample count is : {}'.format(count))
        count += 1
    label1_aug_list = [1] * len(data1_aug_list)

    print('0 count is : {}, 1 count is : {}'.format(len(data0_aug_list), len(data1_aug_list)))

    fea_zero_all_concat = data0_aug_list + data1_aug_list
    fea_zero_label_all = label0_aug_list + label1_aug_list

    # split to train,val,test
    logger.info('start to split data to train,test,valid,according to the split rate')
    random_seed = config['dataConfig']['random_seed']
    split_portion = config['dataSplit']['train_portion'], config['dataSplit']['valid_portion'], config['dataSplit'][
        'test_portion']
    if random_seed is not None:
        idx_list = [i for i in range(len(fea_zero_label_all))]
        random.Random(random_seed).shuffle(idx_list)
        new_feature_list, new_label_list = [], []
        for idx in idx_list:
            new_feature_list.append(fea_zero_all_concat[idx])
            # new_feature_list.append(scaler_std.fit_transform(fea_zero_all_concat[idx]))
            # new_feature_list.append(scaler_minmax.fit_transform(fea_zero_all_concat[idx]))
            new_label_list.append(fea_zero_label_all[idx])
        train_ratio, valid_ratio, test_ratio = split_portion
        valid_start_idx = math.ceil(len(new_label_list) * train_ratio)
        test_start_idx = math.ceil(len(new_label_list) * (train_ratio + valid_ratio))
        train_feature = new_feature_list[: valid_start_idx]
        train_label = new_label_list[: valid_start_idx]
        valid_feature = new_feature_list[valid_start_idx: test_start_idx]
        valid_label = new_label_list[valid_start_idx: test_start_idx]
        test_feature = new_feature_list[test_start_idx:]
        test_label = new_label_list[test_start_idx:]

        logger.info('start to save (data,label) to train,valid,test  ')
        all_data_dict = {
            'train_feature': train_feature,
            'train_label': train_label,
            'valid_feature': valid_feature,
            'valid_label': valid_label,
            'test_feature': test_feature,
            'test_label': test_label
        }

        split_fea_label_path = config['dataPath']['processed_path']
        with open(
                split_fea_label_path + 'winsize100_epilepsy_train_valid_test_fea_label_nostd_16_100_20241210_stride0_20_stride1_1_aug_random1_2w.pkl',
                'wb') as f:
            pickle.dump(all_data_dict, f)


def main():
    FLAGS = get_parse()
    config = yaml.load(open(FLAGS.config, "r"), Loader=yaml.FullLoader)
    get_feature_data(config)


if __name__ == '__main__':
    main()
