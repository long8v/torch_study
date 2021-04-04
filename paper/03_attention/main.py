import os
import argparse
import torch
from sklearn.model_selection import KFold
import torch.optim as optim
from source.dataloader import *
from source.trainer import *
from source.model import *
from tqdm import tqdm
from torch.utils.data.dataset import Subset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'model trainer')
    parser.add_argument('-bs', '--batch_size', type = int, default = 50) 
    parser.add_argument('-epochs', '--n_epochs', type = int, default = 10)   
    parser.add_argument('-src_embd_dim', '--ENC_EMB_DIM', type = int, default = 620)
    parser.add_argument('-trg_embd_dim', '--DEC_EMB_DIM', type = int, default = 620)
    parser.add_argument('-src_hid_dim', '--ENC_HID_DIM', type = int, default = 1000)
    parser.add_argument('-trg_hid_dim', '--DEC_HID_DIM', type = int, default = 1000)
    parser.add_argument('-maxout_hid_dim', '--MAXOUT_HID_DIM', type = int, default = 500)
    parser.add_argument('-maxout_pool_size', '--MAXOUT_POOLSIZE', type = int, default = 2)
    parser.add_argument('-encoder_dropout', '--ENC_DROPOUT', type = int, default = 0.5)
    parser.add_argument('-decoder_dropout', '--DEC_DROPOUT', type = int, default = 0.5)
    parser.add_argument('-save_path', '--save_path', type = str, default = './output')
    parser.add_argument('-model_name', '--model_name', type = str, default = f'basic_{get_today()}')
    parser.add_argument('-teacher_forcing_ratio', '--teacher_forcing_ratio', type = float, default = 0)
    parser.add_argument('-clip', '--clip', type=int, default = 1)
    args = parser.parse_args()
    attention_trainer(args)