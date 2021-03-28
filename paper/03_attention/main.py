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

def main():
    dataset = Multi30k_dataset()
    train_iter, valid_iter, test_iter = Multi30k_iterator(dataset, args.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'your  deivce is {device}')
    
    INPUT_DIM, OUTPUT_DIM = len(dataset.src_field.vocab), len(dataset.trg_field.vocab)

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, args.ENC_EMB_DIM, args.ENC_HID_DIM, args.DEC_HID_DIM, args.ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, args.DEC_EMB_DIM, args.ENC_HID_DIM, args.DEC_HID_DIM, args.DEC_DROPOUT, 
                args.MAXOUT_HID_DIM, args.MAXOUT_POOLSIZE, attn)

    model = Seq2Seq(enc, dec).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = dataset.trg_field.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    logging_time()
    logging_model(model)
    logging_count_parameters(model)
    logging_train(train_iter, valid_iter, optimizer, criterion, args.save_path, 
    args.n_epochs, args.model_name)
    loggint_test(model, test_iter, criterion)

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
    args = parser.parse_args()

    main()