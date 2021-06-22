import torch
import torch.nn as nn
import torch.optim as optim
from .attention import *
from .encoder import *
import random
import math
import time
import pytorch_lightning as pl
    
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
        self.encoder = Encoder(input_dim, config['hid_dim'], 
                               config['n_layers'], config['n_heads'], 
                               config['pf_dim'], config['dropout'],
                            self._device)
        self.nsp = nn.Linear(config['hid_dim'], 2)
        self.lr = self.config['train']['lr']
        self.criterion = nn.CrossEntropyLoss()
        
        
    def make_src_mask(self, src):
        #src = [batch size, src len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]
        return src_mask
    
    def forward(self, src):
        src_mask = self.make_src_mask(src)
        output = self.encoder(src, src_mask)  # batch_size, seq_len, hid_dim
        _cls = output[:, 0, :] # batch_size, hid_dim
        _nsp = self.nsp(_cls) # batch_size, 2
        return _nsp
        
    def training_step(self, batch, batch_nb):
        ids = batch.ids.to(self._device)
        mask_ids = batch.mask_ids.to(self._device)
        replaced_ids = batch.replaced_ids.to(self._device)
        segment_ids = batch.segment_ids.to(self._device)
        nsp = batch.nsp.to(self._device)
        output = self(replaced_ids)
        loss = self.criterion(output, nsp.squeeze(1))
        accuracy = self.multi_acc(output.reshape(-1, 2), nsp.reshape(-1))
        self.log('train_loss', loss, on_step=True)
        self.log('train_accuracy', accuracy, on_step=True)    
        return loss
    
    def validation_step(self, batch, batch_nb):
        ids = batch.ids.to(self._device)
        mask_ids = batch.mask_ids.to(self._device)
        replaced_ids = batch.replaced_ids.to(self._device)
        segment_ids = batch.segment_ids.to(self._device)
        nsp = batch.nsp.to(self._device)
        output = self(replaced_ids)
        loss = self.criterion(output, nsp.squeeze(1))
        accuracy = self.multi_acc(output.reshape(-1, 2), nsp.reshape(-1))
        self.log('valid_loss', loss, on_step=True)
        self.log('valid_accuracy', accuracy, on_step=True)    
        return loss
    
    def multi_acc(self, y_pred, y_test):
        _, y_pred_tags = torch.max(y_pred, dim = 1)
        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc
    
    
    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
        
if __name__ == '__main__':
    input_dim = 100
    hid_dim = 128
    n_layers = 3
    n_heads = 8
    pf_dim = 512
    dropout = 0.5
    device = 'cpu'
    bert = BERT(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device)

    src = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]])
    nsp = torch.tensor([1, 0]).long()
    output = bert(src)
    loss = bert.criterion(output, nsp)
    print(loss)