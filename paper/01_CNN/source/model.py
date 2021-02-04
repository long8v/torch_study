import argparse
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, pretrained_embedding, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx, model_type='multi-channel'):
        
        super().__init__()
        self.static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.nonstatic_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.static_embedding = self.static_embedding.from_pretrained(pretrained_embedding.clone().detach())
        self.nonstatic_embedding = self.nonstatic_embedding.from_pretrained(pretrained_embedding.clone().detach(), 
                                                 max_norm=3.0, freeze=False)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim),
                                              padding = (fs - 1 , 0)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout()
        
    def forward(self, text):

        embedded = self.nonstatic_embedding(text).float()
        embedded_static = self.static_embedding(text).float()
        embedded = embedded.unsqueeze(1)
        embedded_static = embedded_static.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) + F.relu(conv(embedded_static)).squeeze(3) 
                  for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) 
                    for conv in conved] 
        cat = torch.cat(pooled, dim = 1)
        output = self.dropout(cat)
        output = self.fc(output)
        return output