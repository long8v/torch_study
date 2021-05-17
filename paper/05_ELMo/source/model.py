import re
import mecab
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], embedding_dim)) 
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], embedding_dim))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                        out_channels = n_filters, 
                        kernel_size = (filter_sizes[2], embedding_dim))

        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len, token len]
#         print(text.shape) # torch.Size([16, 64, 3])
        bs, max_seq_len, max_chr_len = text.shape
        
        embedded = self.embedding(text.reshape(-1, max_chr_len).unsqueeze(1))
#         print(embedded.shape) # batch_size * max_seq_len, 1, max_chr_len, embed_dim
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
#         print(conved_0.shape, conved_1.shape, conved_2.shape) 
#           torch.Size([1024, 10, 3]) torch.Size([1024, 10, 2]) torch.Size([1024, 10, 1]) 

#         print(conved_0.shape) # [1024, 10, 3]
        pooled_0 = torch.max(conved_0, dim = 1,).values
#         print(pooled_0.shape) # [1024, 3]
        pooled_1 = torch.max(conved_1, dim = 1).values
        pooled_2 = torch.max(conved_2, dim = 1).values
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = -1))
#         print(cat.shape) # 1024, 6
        cat = cat.view(bs, max_seq_len, -1) # 
#         print(cat.shape)
        return cat

# https://github.com/kefirski/pytorch_Highway/blob/master/highway/highway.py
class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x
    

# https://github.com/GyuminJack/torchstudy/blob/main/05May/ELMo/src/models.py
class LSTM_LM(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout, bidirectional = bidirectional)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        input = input.permute(1, 0, 2)
        output, (hidden, cell) = self.lstm(input)  
        seq_len, bs, _ = output.size()
        output = output.reshape(seq_len, bs, -1, 2)
#         print(output.shape) # torch.Size([64, 16, 1024, 2])
        forward_hidden, backward_hidden = output[:,:,:,0], output[:,:,:,1]
        forward_prediction = self.fc_out(forward_hidden).permute(1, 0, 2)
        backward_prediction = self.fc_out(backward_hidden).permute(1, 0, 2)
        return forward_prediction, backward_prediction
    
class ELMo(nn.Module):
    def __init__(self, cnn, highway, rnn):
        super().__init__()
        self.cnn = cnn
        self.highway = highway
        self.rnn = rnn
        
    def forward(self, input):
        output = self.cnn(input)
        output = self.highway(output)
        forward_output, backward_output = self.rnn(output)
        return forward_output, backward_output
    

