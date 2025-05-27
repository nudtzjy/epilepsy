"""
time: 20241125
load the trained epilepsy model  to predict
"""
import numpy as np
import torch
import os
from scipy import io
import shutil
import argparse
from utils import sample_matrix_with_global_indices_std, softmax, sample_matrix_with_global_indices_nostd
from model import get_model_mask
from logger import logger
from torch import FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', help='', default=100, type=int)  # 100
parser.add_argument('--output_dim', help='', default=100, type=int)  # 100
parser.add_argument('--d_model', help='', default=1024, type=int)
parser.add_argument('--n_layer', help='', default=8, type=int)
parser.add_argument('--dim_feedforward', help='', default=1024, type=int)
parser.add_argument('--n_head', help='', default=8)

parser.add_argument('--threshold', help='', default=0.99, type=float)
parser.add_argument('--stride', help='', default=5, type=int)
parser.add_argument('--window_size', help='', default=100, type=int)  # 100

parser.add_argument('--device', help='', default='cuda:0', type=str)
parser.add_argument("--infer_ckpt", type=str,
                    default="data/ckpt/epoch_0_val_acc_0.997500000_model.pth")
parser.add_argument("--infer_input_path", type=str,
                    default="data/infer/PTX_CA3_U20130429_15_ch9-16_convert.mat")
args = vars(parser.parse_args())
for arg in args:
    logger.info('{}: {}'.format(arg, args[arg]))


def main():
    device = args['device']
    # transformer parameter
    input_dim = args['input_dim']
    output_dim = args['output_dim']
    d_model = args['d_model']
    n_layer = args['n_layer']
    dim_feedforward = args['dim_feedforward']
    n_head = args['n_head']
    class_num = 2

    model = get_model_mask(input_dim, output_dim, dim_feedforward, d_model,
                           n_head, n_layer, class_num, device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has parameters: {n_parameters / 1e6}M")

    # load the trained model
    load_params = torch.load(args['infer_ckpt'], map_location=device)
    model.load_state_dict(load_params)
    print("Parameters successfully loaded.")
    model.eval()

    infer_input_data = args['infer_input_path']
    mat_data = io.loadmat(infer_input_data)
    mat_data_val = mat_data['data_convert']
    data = np.array(mat_data_val)

    window_size = args['window_size']
    stride = args['stride']

    # samples_with_indices = sample_matrix_with_global_indices(data, window_size, stride)
    # add the infer input data with std preprocess before input model
    # samples_with_indices = sample_matrix_with_global_indices_std(data, window_size, stride)

    samples_with_indices = sample_matrix_with_global_indices_nostd(data, window_size, stride)
    for thred in [0.999]:
        epilepsy_index = []
        for sample, index in samples_with_indices:
            input_src = FloatTensor(sample).to(device)
            predict = model(input_src)
            predict = predict.detach().to('cpu').numpy()
            predict = predict[0, :]
            predict_softmax = softmax(np.array(predict))
            predict_max = np.max(predict_softmax)
            predict_idx = np.argmax(predict_softmax)

            if predict_max >= thred and predict_idx == 1:
                print('Detected epilepsy, start index: {}'.format(index))
                epilepsy_index.append(index)

        save_path_dir = os.path.dirname(infer_input_data)
        outputdata_dir_name = infer_input_data.strip().split('/')[-1].split('.')[0]
        file_name = outputdata_dir_name
        save_name = save_path_dir + '/' + '{}_thred_{}_ws_{}_stride_{}_nostd.txt'.format(file_name,
                                                                                         thred, window_size, stride)
        output_path = save_path_dir + '/' + outputdata_dir_name
        os.makedirs(output_path, exist_ok=True)

        np.savetxt(save_name, epilepsy_index, fmt='%d')
        shutil.move(save_name, output_path)

    logger.info('inference complete')


if __name__ == '__main__':
    main()
