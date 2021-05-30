
import pytorch_lightning as pl
import re
import mecab
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score

class simpleGRU_model(pl.LightningModule):
    def __init__(self, config, input_dim, embedding_dim, n_layers, hid_dim, output_dim):
        super(simpleGRU_model, self).__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hid_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        bs = x.shape[0]
        output = self.embedding(x)
        output = output.reshape(bs, -1, self.embedding_dim)
        _, output = self.gru(output)
        output = self.fc(output[-1, :, :])
        return output
    
    def training_step(self, batch, batch_nb):
        src_chr, trg = batch
        src_chr, trg = src_chr.to(self.device), trg.to(self.device)
        output = self(src_chr)
        loss = self.criterion(output, trg)
        accuracy = self.multi_acc(output, trg)
        fscore = self.fscore(output, trg)
        self.log('train_loss', loss, on_step=True)
        self.log('train_accuracy', accuracy, on_step=True)
        self.log('train_fscore', fscore, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        src_chr, trg = batch
        src_chr, trg = src_chr.to(self.device), trg.to(self.device)
        output = self(src_chr)
        loss = self.criterion(output, trg)
        accuracy = self.multi_acc(output, trg)
        fscore = self.fscore(output, trg)
        self.log('valid_loss', loss, on_step=True)
        self.log('valid_accuracy', accuracy, on_step=True)
        self.log('valid_fscore', fscore, on_step=True)
        return loss
    
    def multi_acc(self, y_pred, y_test):
        _, y_pred_tags = torch.max(y_pred, dim = 1)
        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc
    
    def fscore(self, y_pred, y_test):
        _, y_pred_tags = torch.max(y_pred, dim = 1)
        return f1_score(y_test.cpu(), y_pred_tags.cpu(), average='macro')

        
    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.config['TRAIN']['LR'])
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
    

class simpleGRU_model_w_elmo(pl.LightningModule):
    def __init__(self, config, elmo, input_dim, embedding_dim, max_chr_len, n_layers, hid_dim, output_dim):
        super(simpleGRU_model_w_elmo, self).__init__()
        self.config = config
        self.elmo = elmo
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim * (max_chr_len + 2), hid_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.n_layers = n_layers
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        
        ### get_ elmo vector ###
        # x : batch_size, max_token_len, max_chr_len
        bs = x.shape[0]
        with torch.no_grad():
            forward_hidden, backward_hidden = self.elmo.forward(x, finetune=True)
        # forward_hidden : seq_len, batch, hidden_size
        elmo_vector = torch.stack([forward_hidden, backward_hidden])   
        
        ### another rnn ### 
        # elmo_vector : 2, seq_len, batch, hidden_dim
        output = self.embedding(x)
        # output : batch_size, seq_len, max_chr_len, embedding_dim
        elmo_vector = elmo_vector.permute(2, 1, 0, 3)
        output = torch.cat([elmo_vector, output], dim = 2)
        # output : batch_size ,seq_len, max_chr_len + 2, embedding_dim
        seq_len =  output.shape[1]
        # output : batch_size, seq_len, (max_chr_len + 2) * embeding_dim
        output = output.reshape(bs, seq_len, -1)
        _, output = self.gru(output) # (num_layers * num_directions, batch, hidden_size)
        output = output.transpose(1, 0)
        # (batch, num_layers * num_directions, hidden_size)
        output = self.fc(output[:, -1, :]) # batch_first
        return output
    
    def training_step(self, batch, batch_nb):
        src_chr, trg = batch
        src_chr, trg = src_chr.to(self.device), trg.to(self.device)
        output = self(src_chr)
        loss = self.criterion(output, trg)
        accuracy = self.multi_acc(output, trg)
        fscore = self.fscore(output, trg)
        self.log('train_loss', loss, on_step=True)
        self.log('train_accuracy', accuracy, on_step=True)
        self.log('train_fscore', fscore, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        src_chr, trg = batch
        src_chr, trg = src_chr.to(self.device), trg.to(self.device)
        output = self(src_chr)
        loss = self.criterion(output, trg)
        accuracy = self.multi_acc(output, trg)
        fscore = self.fscore(output, trg)
        self.log('valid_loss', loss, on_step=True)
        self.log('valid_accuracy', accuracy, on_step=True)
        self.log('valid_fscore', fscore, on_step=True)
        return loss
    
    def multi_acc(self, y_pred, y_test):
        _, y_pred_tags = torch.max(y_pred, dim = 1)
        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc
    
    def fscore(self, y_pred, y_test):
        _, y_pred_tags = torch.max(y_pred, dim = 1)
        return f1_score(y_test.cpu(), y_pred_tags.cpu(), average='macro')

        
    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.config['TRAIN']['LR'])
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
    