import torch
import torch.nn as nn
import sys
sys.path.append('/home/long8v/torch_study/paper/06_BERT/source/')
from model.attention import *
import random
import math
import time
    
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, # vocabulary 개수
                 hid_dim,   # token의 임베딩 차원
                 max_len,
                 n_layers,  # self-attention + FCN 레이어를 몇 층 쌓을건지
                 n_heads,   # 몇 개의 multi-head self-attention
                 pf_dim,    # FCN의 dimension
                 dropout,  
                 device,
                 max_length = 100): 
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim + 10, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)
        self.seg_embedding = nn.Embedding(3, hid_dim) # senA, senB, padding
#         self.layer_norm = nn.LayerNorm(hid_dim) # 성능이 더 안좋아짐
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, seg, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #     (src_len)              (1, src_len)  (batch_size, src_len)  
        
        
        #pos = [batch size, src len]
        src = self.tok_embedding(src) + self.pos_embedding(pos) + self.seg_embedding(seg)
#         src = self.layer_norm(src)
        
        for layer in self.layers:
            src = layer(src, src_mask)  # batch_size, seq_len, hid_dim
        
        return src 
    
    
if __name__ == '__main__':
    input_dim = 100
    hid_dim = 128
    n_layers = 3
    n_heads = 8
    pf_dim = 512
    dropout = 0.5
    device = 'cpu'
    enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device)

    src = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]])
    bs, src_len = src.shape
    src_mask = torch.zeros(bs, src_len).unsqueeze(1).unsqueeze(2).long()
    output = enc(src, src_mask)
    print(output.shape)
