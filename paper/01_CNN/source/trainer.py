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

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

from .model import *
from .dataloader import *

# writer = SummaryWriter('runs/cnn')
SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        src = batch[0].to(device) 
        trg = torch.Tensor(batch[1]).to(device).long()
        optimizer.zero_grad()
        predictions = model(src)
        loss = criterion(predictions, trg)
        
        # ## l2 weight norm https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        # l2_lambda = 0.01
        # l2_reg = torch.tensor(0., requires_grad=False)
        # for param in model.parameters():
        #     l2_reg.data.add_(torch.sqrt(torch.norm(param)))
        #     loss.add_(l2_lambda * l2_reg)

        acc = softmax_accuracy(predictions, trg)
        loss.backward()
        optimizer.step()

                
        ## max-norm 
        for name, param in model.named_parameters():
            max_val = 3
            eps = 1e-12
            if 'fc.weight' in name:
                norm = torch.norm(param, 2, dim=0)
                desired = torch.clamp(norm, 0, max_val)
                param.data *= (desired / (eps + norm))
        
    
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            src = batch[0].to(device) 
            trg = torch.Tensor(batch[1]).to(device).long()
            predictions = model(src).squeeze(1)
            loss = criterion(predictions, trg)
            acc = softmax_accuracy(predictions, trg)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def softmax_accuracy(preds, y):
    argmax = torch.argmax(preds, dim=1)
    correct = (argmax == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def verbose_train(model, train_dl, valid_dl, optimizer, criterion, save_path, N_EPOCHS):
    best_valid_accuracy = float(0)
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_dl, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_dl, criterion)
        # writer.add_scalar('training loss', train_loss, epoch)
        # writer.add_scalar('training acc', train_acc, epoch)
        # writer.add_scalar('valid loss', valid_loss, epoch)
        # writer.add_scalar('valid acc', valid_acc, epoch)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_acc > best_valid_accuracy:
            best_valid_accuracy = valid_acc
            torch.save(model.state_dict(), f'{save_path}/torch_study/data/tut4-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}\t| Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f}\t|  Val. Acc: {valid_acc*100:.2f}%')

            