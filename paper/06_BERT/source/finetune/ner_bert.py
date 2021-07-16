import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append('/home/long8v/torch_study/paper/06_BERT/source/')
from model.attention import *
from model.encoder import *
from model.bert import *
from utils import *
import random
import math
import time
import pytorch_lightning as pl
import yaml
from torchcrf import CRF


class NER_BERT(pl.LightningModule):
    def __init__(self, 
                 config,
                 input_dim,
                 output_dim,
                 pad_idx):
        super().__init__()
        self.config = config
        self.pad_idx  = pad_idx
        config = self.config['model']
        self._device = self.config['train']['device']
        pretrained_path = self.config['model']['pretrained_path']
        print(f'load pretrained model from {pretrained_path}..')
        pretrained_config = read_yaml(f'{pretrained_path}/model_config.yaml')
        pretrained_model = f'{pretrained_path}/model.pt'
        self.bert = BERT(self.config, pretrained_config['data']['input_dim'],
                        pretrained_config['data']['pad_idx'])
        self.bert.load_state_dict(torch.load(pretrained_model))
        self.lr = self.config['train']['lr']
        self.encoder = self.bert.encoder
        self.output_dim = output_dim
        self.fcn = nn.Linear(config['hid_dim'], output_dim)
        self.crf = CRF(output_dim, batch_first=True)
        self.criteiron = nn.CrossEntropyLoss(ignore_index=pretrained_config['data']['pad_idx'])
        
    def make_src_mask(self, src):
        #src = [batch size, src len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]
        return src_mask
    
    def forward(self, tokens, labels):
        seg = torch.zeros_like(tokens) # seg는 아무래도 상관없음
        token_mask = self.make_src_mask(tokens)
        output = self.encoder(tokens, seg, token_mask)  # batch_size, seq_len, hid_dim
        output = self.fcn(output)
        loss = self.crf(output, labels)
        output = torch.tensor(self.crf.decode(output))
        return loss, output
        
    def training_step(self, batch, batch_nb):
        token = batch.token.to(self._device)
        label = batch.label.to(self._device)
        loss, output = self(token, label)
        accuracy = self.acc(output.to(self._device), label)
        self.log('train_loss', loss, on_step=True)
        self.log('train_accuracy', accuracy, on_step=True)
        self.log('lr', self.optimizer.param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_nb):
        token = batch.token.to(self._device)
        label = batch.label.to(self._device)
        loss, output = self(token, label)
        accuracy = self.acc(output.to(self._device), label)
        self.log('valid_loss', loss, on_step=True)
        self.log('valid_accuracy', accuracy, on_step=True)
        self.log('lr', self.optimizer.param_groups[0]['lr'])
        return loss
        
    def acc(self, y_pred, y_test):
        correct_pred = (y_pred == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc
        
    def multi_acc(self, y_pred, y_test):
        if y_pred.numel():
            _, y_pred_tags = torch.max(y_pred, dim = 1)
            correct_pred = (y_pred_tags == y_test).float()
            acc = correct_pred.sum() / len(correct_pred)
            acc = torch.round(acc * 100)
            return acc
        return 0
    
    
    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))
        
    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr = self.lr)
        if self.config['train']['scheduler']:
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
            return  {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "valid_loss"}
        return self.optimizer
        
if __name__ == '__main__':
    input_dim = 100
    hid_dim = 128
    n_layers = 3
    n_heads = 8
    pf_dim = 512
    dropout = 0.5
    output_dim = 13 
    device = 'cpu'
    config_file = '/home/long8v/torch_study/paper/06_BERT/config_finetune.yaml'
    config = yaml.safe_load(open(config_file, 'r', encoding='utf8'))
    bert = NER_BERT(config, input_dim, output_dim, 0)
    bert.to(device)
    