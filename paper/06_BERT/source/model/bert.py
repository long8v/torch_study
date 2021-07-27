import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR
from .attention import *
from .encoder import *
import random
import math
import time
import pytorch_lightning as pl
import yaml


class BERT(pl.LightningModule):
    def __init__(self, 
                 config,
                 input_dim,
                 pad_idx):
        super().__init__()
        self.config = config
        self.pad_idx  = pad_idx
        config = self.config['model']
        self._device = self.config['train']['device']
        self.encoder = Encoder(input_dim, config['hid_dim'], self.config['data']['max_len'],
                               config['n_layers'], config['n_heads'], 
                               config['pf_dim'], config['dropout'],
                            self._device)
        self.nsp = nn.Linear(config['hid_dim'], 2)
        self.mlm = nn.Linear(config['hid_dim'], input_dim)
        self.train_nsp = self.config['train']['train_nsp']
        self.train_mlm = self.config['train']['train_mlm']
        self.lr = self.config['train']['lr']
        self.criterion_nsp = nn.CrossEntropyLoss()
        self.criterion_mlm = nn.CrossEntropyLoss(ignore_index = 0) # 하드코딩 나중에 수정 
        
    def make_src_mask(self, src):
        #src = [batch size, src len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]
        return src_mask
    
    def forward(self, src, seg):
        src_mask = self.make_src_mask(src)
        output = self.encoder(src, seg, src_mask)  # batch_size, seq_len, hid_dim
        _cls = output[:, 0, :] # batch_size, hid_dim
        _nsp = self.nsp(_cls) # batch_size, 2
        _mlm = self.mlm(output)
        return _nsp, _mlm
        
    def training_step(self, batch, batch_nb):
        ids = batch.ids.to(self._device)
        mask_ids = batch.mask_ids.to(self._device)
        replaced_ids = batch.replaced_ids.to(self._device)
        segment_ids = batch.segment_ids.to(self._device)
        nsp = batch.nsp.to(self._device)
        nsp_output, mlm_output = self(replaced_ids, segment_ids)
        total_loss = 0 
        if self.train_nsp:
            nsp_loss = self.criterion_nsp(nsp_output, nsp.squeeze(1))
            nsp_accuracy = self.multi_acc(nsp_output.reshape(-1, 2), nsp.reshape(-1))
            total_loss += nsp_loss
            self.log('train_nsp_loss', nsp_loss, on_step=True)
            self.log('train_nsp_accuracy', nsp_accuracy, on_step=True)
        if self.train_mlm:
            tmpmask = torch.zeros_like(mlm_output)
            masked_mlm = mlm_output[mask_ids.bool()]
            target_mlm = torch.masked_select(ids, mask_ids == 1)
            mlm_loss = self.criterion_mlm(masked_mlm, target_mlm)        
            mlm_accuracy = self.multi_acc(masked_mlm, target_mlm.reshape(-1))
            total_loss += mlm_loss
            self.log('train_mlm_loss', mlm_loss, on_step=True)
            self.log('train_mlm_accuracy', mlm_accuracy, on_step=True)
            
        self.log('train_total_loss', total_loss, on_step=True)
        self.log('lr', self.optimizer.param_groups[0]['lr'])
        return total_loss

    def validation_step(self, batch, batch_nb):
        ids = batch.ids.to(self._device)
        mask_ids = batch.mask_ids.to(self._device)
        replaced_ids = batch.replaced_ids.to(self._device)
        segment_ids = batch.segment_ids.to(self._device)
        nsp = batch.nsp.to(self._device)
        nsp_output, mlm_output = self(replaced_ids, segment_ids)
        if self.train_nsp:
            nsp_loss = self.criterion_nsp(nsp_output, nsp.squeeze(1))
            nsp_accuracy = self.multi_acc(nsp_output.reshape(-1, 2), nsp.reshape(-1))
            total_loss += nsp_loss
            self.log('valid_nsp_loss', nsp_loss, on_step=True)
            self.log('valid_nsp_accuracy', nsp_accuracy, on_step=True)
        if self.train_mlm:
            tmpmask = torch.zeros_like(mlm_output)
            masked_mlm = mlm_output[mask_ids.bool()]
            target_mlm = torch.masked_select(ids, mask_ids == 1)
            mlm_loss = self.criterion_mlm(masked_mlm, target_mlm)        
            mlm_accuracy = self.multi_acc(masked_mlm, target_mlm.reshape(-1))
            total_loss += mlm_loss
            self.log('valid_mlm_loss', mlm_loss, on_step=True)
            self.log('valid_mlm_accuracy', mlm_accuracy, on_step=True)
        self.log('valid_total_loss', total_loss, on_step=True)
        self.log('lr', self.optimizer.param_groups[0]['lr'])    
        return total_loss

        
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
#             self.scheduler = WarmupConstantSchedule(self.optimizer, d_model=self.config['model']['hid_dim'],
#                                                warmup_steps=self.config['train']['warmup_steps'])
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
#             self.scheduler = CosineAnnealingLR(self.optimizer, T_max=5)
            return  {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "valid_total_loss"}
        return self.optimizer
        
class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, d_model, warmup_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return 1
            lrate = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
            return lrate
        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda)
        
if __name__ == '__main__':
    input_dim = 100
    hid_dim = 128
    n_layers = 3
    n_heads = 8
    pf_dim = 512
    dropout = 0.5
    output_dim = 100
    device = 'cpu'
    config_file = '/home/long8v/torch_study/paper/06_BERT/config.yaml'
    config = yaml.safe_load(open(config_file, 'r', encoding='utf8'))
    bert = BERT(config, input_dim, output_dim, 0)
    bert.to(device)

    src = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]])
    nsp = torch.tensor([1, 0]).long()
    seg = torch.tensor([[0, 0, 1, 1], [0, 0,0 ,1]])
    output, output2 = bert(src.to(device), seg.to(device))
    
    loss = bert.criterion(output, nsp)
    print(loss)