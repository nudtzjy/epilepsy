"""
time: 20241204
build the transformer model to epilepsy detect
"""
import torch
import pickle
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
import argparse
from utils import ModelDataset
from model import get_model_mask, model_eval_mask, loss_calculate_mask
from logger import logger
from torch.optim.lr_scheduler import StepLR

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', help='', default=100, type=int)
parser.add_argument('--output_dim', help='', default=100, type=int)
parser.add_argument('--d_model', help='', default=1024, type=int)
parser.add_argument('--n_layer', help='', default=8, type=int)
parser.add_argument('--dim_feedforward', help='', default=1024, type=int)
parser.add_argument('--n_head', help='', default=8)

parser.add_argument('--max_epoch', help='', default=200, type=int)
parser.add_argument('--step_size', help='', default=100, type=int)
parser.add_argument('--gamma', help='', default=0.1, type=float)
parser.add_argument('--learning_rate', help='', default=0.00001, type=float)
parser.add_argument('--batch_size', help='', default=2048, type=int)
parser.add_argument('--device', help='', default='cuda:1', type=str)
parser.add_argument("--finetune", action="store_true", default=False)
parser.add_argument("--finetune_ckpt", type=str,
                    default="data/ckpt/xx_model.pth")

parser.add_argument("--ckpt_folder", type=str,
                    default="data/ckpt/")
parser.add_argument("--model_name", type=str,
                    default="episy_tfs_1024_1e4")
parser.add_argument("--input_feature_path", type=str,
                    default="data/processed/winsize100_epilepsy_train_valid_test_fea_label_nostd_16_100_20241210_stride0_30_stride1_1_aug_random1_2w.pkl")
args = vars(parser.parse_args())
for arg in args:
    logger.info('{}: {}'.format(arg, args[arg]))


def main():
    max_epoch = args['max_epoch']
    batch_size = args['batch_size']
    device = args['device']
    learning_rate = args['learning_rate']
    label_type = 'discrete'
    # transformer parameter
    input_dim = args['input_dim']
    output_dim = args['output_dim']
    d_model = args['d_model']
    n_layer = args['n_layer']
    dim_feedforward = args['dim_feedforward']
    n_head = args['n_head']
    class_num = 2

    ckpt_folder_real = args['ckpt_folder'] + args['model_name']
    os.makedirs(ckpt_folder_real, exist_ok=True)
    # save config para
    output_file = ckpt_folder_real + '/' + 'cmd.txt'
    with open(output_file, 'w') as file:
        for arg in args:
            file.write(arg + ' : ' + str(args[arg]) + '\n')

    summary_writer_logs = ckpt_folder_real + '/logs/'
    os.makedirs(summary_writer_logs, exist_ok=True)
    writer = SummaryWriter(log_dir=summary_writer_logs)

    # load the saved  train,valid,test (feature ,label) dataset
    with open(args['input_feature_path'], 'rb') as f:
        all_data_dict = pickle.load(f)

    train_dataset = ModelDataset(all_data_dict, is_train=True, is_test=False)
    val_dataset = ModelDataset(all_data_dict, is_train=False, is_test=False)
    test_dataset = ModelDataset(all_data_dict, is_train=False, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=8, drop_last=False, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=8, drop_last=False, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=8, drop_last=False, shuffle=False)

    model = get_model_mask(input_dim, output_dim, dim_feedforward, d_model,
                           n_head, n_layer, class_num, device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has parameters: {n_parameters / 1e6}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=args['step_size'], gamma=0.1)

    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    if args['finetune']:
        load_params = torch.load(args['finetune_ckpt'], map_location=device)
        model.load_state_dict(load_params)

    logger.info('initial evaluation')
    model.eval()
    valid_mae = model_eval_mask(model, val_loader, batch_size, label_type, device)
    logger.info('initial valid metric: {}'.format(valid_mae))

    logger.info('start training')
    best_val_acc = 0.0
    for epoch in range(max_epoch):
        logger.info('start epoch: {}'.format(epoch))
        train_avg_loss, train_count, valid_avg_loss, valid_count = 0, 0, 0, 0
        model.train()
        for batch_idx, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            loss = loss_calculate_mask(batch, device, model, loss_func, label_type)
            loss.backward()
            optimizer.step()
            train_avg_loss += loss.detach()
            train_count += 1
            logger.info(
                'epoch:{}, step:{}/{} ,lr :{:.6f},train batch loss: {}'.format(epoch, batch_idx, len(train_loader),
                                                                               scheduler.get_last_lr()[0],
                                                                               loss.detach()))
        scheduler.step()
        model.eval()
        # for batch in val_loader:
        for batch_idx, batch in enumerate(val_loader, 0):
            loss = loss_calculate_mask(batch, device, model, loss_func, label_type)
            valid_count += 1
            valid_avg_loss += loss.detach()
        val_acc = model_eval_mask(model, val_loader, batch_size, label_type, device)
        logger.info('epoch {}, train avg loss: {},valid avg loss: {}'
                    .format(epoch, train_avg_loss / train_count, valid_avg_loss / valid_count))
        logger.info('epoch: {}, valid acc: {}'.format(epoch, val_acc))
        writer.add_scalar('trainLoss: ', train_avg_loss / train_count, global_step=epoch)
        writer.add_scalar('validLoss: ', valid_avg_loss / valid_count, global_step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_folder_real,
                                                        'epoch_{}_val_acc_{:.9f}_model.pth'.format(epoch,
                                                                                                   val_acc)))
            logger.info('epoch: {}, the best val acc is : {}'.format(epoch, val_acc))
    logger.info('training complete')


if __name__ == '__main__':
    main()
