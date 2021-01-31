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
        
        # we experiment with having two 'channels' of word vector
        # ... each filter is applied to calculate c_i
        # ... and the results are added to cacluate c_i
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
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        conved_static = [F.relu(conv(embedded_static)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] 
        pooled_static = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_static] 
        #pooled_n = [batch size, n_filters]
        cat = torch.cat(pooled, dim = 1)
        cat_static = torch.cat(pooled_static, dim = 1)
        cat.add_(cat_static)
        #cat = [batch size, n_filters * len(filter_sizes)] 
        output = self.dropout(cat)
        output = self.fc(output)

        # ouput [batch size, output_dim]
#         output = self.softmax(output) # loss를 CrossEntropyLoss를 쓰니까 생략하자!
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Model Builder')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5])
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CNN(vocab_size=1000, pad_idx=0, args=args).to(device)
    sample = torch.randint(20, (3, 5)).to(device)
    res = model(sample)

    print(res.shape)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')