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
        loss = self.criterion(self(src_chr), trg)
        self.log('train_loss', loss, on_step=True)
        return loss

        
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