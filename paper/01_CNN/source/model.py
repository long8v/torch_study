import argparse
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, pretrained_embedding, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        self.static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.nonstatic_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.static_embedding.from_pretrained(pretrained_embedding.clone().detach())
        self.nonstatic_embedding.from_pretrained(pretrained_embedding.clone().detach(), 
                                                 max_norm=3.0, freeze=False)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout()
#         self.softmax = nn.Softmax(dim=1)
        
    def forward(self, text):
        # text = [batch size, sent len]
        ## static embedding 
        embedded = self.nonstatic_embedding(text)
        embedded_static = self.static_embedding(text)
#         print(f'|embedded_shape| {embedded.shape}')
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        embedded_static = embedded_static.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) + F.relu(conv(embedded_static)).squeeze(3) 
                  for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] 
        cat = torch.cat(pooled, dim = 1)
        #cat = [batch size, n_filters * len(filter_sizes)] 
        output = self.dropout(cat)
        output = self.fc(output)
        # ouput [batch size, output_dim]
        return output