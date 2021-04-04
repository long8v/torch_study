import time
import os
import os.path as osp
import re
import pickle
import argparse
import random
from tqdm import tqdm
import numpy as np
import functools
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from .model import *
from .dataloader import *
from .utils import *

# writer = SummaryWriter('runs/cnn')
SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class attention_trainer:
    def __init__(self, args):
        self.args = args
        self.get_device()

        print('data loading..')
        dataset = Multi30k_dataset()
        self.trg_field = dataset.trg_field
        iterators = Multi30k_iterator(dataset, self.args.batch_size)
        train_iter, valid_iter, test_iter = iterators.train_iter, iterators.valid_iter, iterators.test_iter
        self.INPUT_DIM, self.OUTPUT_DIM = len(dataset.src_field.vocab), len(dataset.trg_field.vocab)
        
        print('model loading')
        model = self.get_seq2seq_model()
        model = model.to(self.device)
        print(f'model is in {model.device}')

        print('training..')
        optimizer = optim.Adam(model.parameters())
        TRG_PAD_IDX = dataset.trg_field.vocab.stoi[dataset.trg_field.pad_token]
        criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

        logging_time()
        logging_model(model)
        logging_count_parameters(model)
        self.logging_train(model, train_iter, valid_iter, optimizer, criterion, self.args.save_path, 
                        self.args.n_epochs, self.args.model_name, self.args.clip, self.args.teacher_forcing_ratio)
        self.logging_test(model, test_iter, criterion)

    def get_seq2seq_model(self):
        attn = Attention(self.args.ENC_HID_DIM, self.args.DEC_HID_DIM)
        enc = Encoder(self.INPUT_DIM, self.args.ENC_EMB_DIM, self.args.ENC_HID_DIM, self.args.DEC_HID_DIM, self.args.ENC_DROPOUT)
        dec = Decoder(self.OUTPUT_DIM, self.args.DEC_EMB_DIM, self.args.ENC_HID_DIM, self.args.DEC_HID_DIM, self.args.DEC_DROPOUT, 
                    self.args.MAXOUT_HID_DIM, self.args.MAXOUT_POOLSIZE, attn)
        model = Seq2Seq(enc, dec, self.args.teacher_forcing_ratio, self.device).to(self.device)
        model.apply(self.init_weights)
        return model

    def get_device(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'your  deivce is {self.device}')


    def init_weights(self, m):
        for name, param in m.named_parameters():
            if 'rnn.weight' in name:
                # recurrent weight matrix 
                # orthogonal should have dim 2 or more
                nn.init.orthogonal_(param.data)
            elif 'attention.v' in name:
                nn.init.zeros_(param.data)
            elif 'attention' in name:
                nn.init.normal_(param.data, 0, 0.001 ** 2)
            elif 'weight' in name:
                nn.init.normal_(param.data, 0, 0.01 ** 2)
            else:
                nn.init.constant_(param.data, 0)
                

    def train(self, model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):
        
        model.train()
        
        epoch_loss = 0
        epoch_bleu = 0 
        
        for i, batch in enumerate(tqdm(iterator)):
            
            src = batch.src
            trg = batch.trg

            src, trg = src.to(self.device), trg.to(self.device)
            
            optimizer.zero_grad()
            
            output = model(src, trg, teacher_forcing_ratio)
            
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            
            bleu = get_bleu_score(output, trg, self.trg_field)
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            
            loss = criterion(output, trg)

            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_bleu += bleu
            
        return epoch_loss / len(iterator), epoch_bleu / len(iterator)

    def evaluate(self, model, iterator, criterion):

        model.eval()
        
        epoch_loss = 0
        epoch_bleu = 0 
        
        with torch.no_grad():
        
            for i, batch in enumerate(iterator):

                src = batch.src
                trg = batch.trg

                src, trg = src.to(self.device), trg.to(self.device)
                output = model(src, trg, 0) #turn off teacher forcing
                
                #trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]
                bleu = get_bleu_score(output, trg, self.trg_field)
                output_dim = output.shape[-1]
                
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                #trg = [(trg len - 1) * batch size]
                #output = [(trg len - 1) * batch size, output dim]

                loss = criterion(output, trg)


                epoch_loss += loss.item()
                epoch_bleu += bleu
            
        return epoch_loss / len(iterator), epoch_bleu / len(iterator)

    def logging_train(self, model, train_dl, valid_dl, optimizer, criterion, save_path, 
                        N_EPOCHS, model_name, clip, teacher_forcing_ratio):
        logging.basicConfig(filename=f'{save_path}/{model_name}.log', level=logging.DEBUG)
        best_valid_bleu = float(0)
        for epoch in range(N_EPOCHS):

            start_time = time.time()
            
            train_loss, train_bleu = self.train(model, train_dl, optimizer, criterion, clip, teacher_forcing_ratio)
            valid_loss, valid_bleu = self.evaluate(model, valid_dl, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_bleu > best_valid_bleu:
                best_valid_bleu = valid_bleu
                torch.save(model.state_dict(), f'{save_path}/{model_name}.pt')

            logging.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            logging.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | train BLEU : {train_bleu:.3f}')
            logging.info(f'\tVal. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | valid BLEU : {valid_bleu:.3f}')
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logging.info(f'\t best Val. accuracy: {best_valid_bleu :.3f}')

    def logging_test(self, model, test_dl, criterion):
        test_loss, test_bleu = self.evaluate(model, test_dl, criterion)
        logging.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU : {test_bleu :.3f}')