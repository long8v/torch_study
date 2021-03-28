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
from sacrebleu import corpus_bleu, sentence_bleu

# writer = SummaryWriter('runs/cnn')
SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_weights(m):
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
            

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    epoch_bleu = 0 
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        bleu = get_bleu_score(output, trg, TRG)
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

def evaluate(model, iterator, criterion):
        
    def get_speical_token(field):
        def get_stoi(idx):
            return field.vocab.stoi[idx]
        return [get_stoi(field.pad_token), get_stoi(field.unk_token), 
                get_stoi(field.eos_token)]

    def get_itos_str(tokens, field):
        ignore_idx = get_speical_token(field)
        return ' '.join([field.vocab.itos[token] for token in tokens
                        if token not in ignore_idx])

    def get_itos_batch(tokens_batch, field):
        return [get_itos_str(batch, field) for batch in tokens_batch]

    def get_bleu_score(output, trg, trg_field):
        with torch.no_grad():
            output_token = output.argmax(-1)
        # 문장 별로 해야돼서 permute 해야 함
        output_token = output_token.permute(1, 0)
        trg = trg.permute(1, 0)
        system = get_itos_batch(output_token, trg_field)
        refs = get_itos_batch(trg, trg_field)
        bleu = corpus_bleu(system, [refs], force=True).score
        return bleu

    model.eval()
    
    epoch_loss = 0
    epoch_bleu = 0 
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            bleu = get_bleu_score(output, trg, TRG)
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)


            epoch_loss += loss.item()
            epoch_bleu += bleu
        
    return epoch_loss / len(iterator), epoch_bleu / len(iterator)

def logging_train(model, train_dl, valid_dl, optimizer, criterion, save_path, N_EPOCHS, model_name):
    logging.basicConfig(filename=f'{model_name}.log', level=logging.DEBUG)
    best_valid_bleu = float(0)
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_bleu = train(model, train_dl, optimizer, criterion)
        valid_loss, valid_bleu = evaluate(model, valid_dl, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_bleu > best_valid_bleu:
            best_valid_bleu = valid_bleu
            torch.save(model.state_dict(), f'{save_path}/{model_name}.pt')

        logging.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logging.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | train BLEU : {train_bleu:.3f}')
        logging.info(f'\tVal. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | valid BLEU : {valid_bleu:.3f}')

    logging.info(f'\t best Val. accuracy: {best_valid_bleu :.3f}')

def logging_test(model, test_dl, criterion):
    test_loss, test_bleu = evaluate(model, test_dl, criterion)
    logging.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU : {test_bleu :.3f}')