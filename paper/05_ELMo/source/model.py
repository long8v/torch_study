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


class CNN(pl.LightningModule):
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
        
#         self.ln = nn.LayerNorm(sum([j for _, j in self.kernal_out_dims]))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len, token len]
        bs, max_seq_len, max_chr_len = text.shape
        
        embedded = self.embedding(text.reshape(-1, max_chr_len).unsqueeze(1))
#         print(embedded.shape) # batch_size * max_seq_len, 1, max_chr_len, embed_dim
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        pooled_0 = torch.max(conved_0, dim = -1).values
        pooled_1 = torch.max(conved_1, dim = -1).values
        pooled_2 = torch.max(conved_2, dim = -1).values
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = -1))
        cat = cat.view(bs, max_seq_len, -1) 
        return cat

# https://github.com/kefirski/pytorch_Highway/blob/master/highway/highway.py
class Highway(pl.LightningModule):
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
class LSTM_LM(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout, bidirectional = bidirectional)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, finetune=False):
        input = input.permute(1, 0, 2)
        output, (hidden, cell) = self.lstm(input)  
        seq_len, bs, _ = output.size()
        # output : (seq_len, batch, num_directions * hidden size) -> (seq_len, batch, hidden_size, num_directions)
        output = output.reshape(seq_len, bs, -1, 2) # 2 because bidirectional, stacked RNN output is last layer output
        forward_hidden, backward_hidden = output[:,:,:,0], output[:,:,:,1]
        # forward_hidden : (seq_len, batch, hidden_size)
        if finetune:
            return forward_hidden, backward_hidden
        # forward_prediction : (seq_len, batch, output_dim) -> (batch, seq_len, output_dim)
        forward_prediction = self.fc_out(forward_hidden).permute(1, 0, 2)
        backward_prediction = self.fc_out(backward_hidden).permute(1, 0, 2)
        return forward_prediction, backward_prediction
    

class ELMo(pl.LightningModule):
    # https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html 
    def __init__(self, config, chr_vocab_size, chr_pad_idx, trg_pad_idx, predict_dim):
        super(ELMo, self).__init__()
        self.config = config
        cnn_config = self.config['MODEL']['CNN']
        self.cnn = CNN(chr_vocab_size, cnn_config['EMBEDDING_DIM'],
                       cnn_config['N_FILTERS'], cnn_config['FILTER_SIZES'], 
                       cnn_config['OUTPUT_DIM'], cnn_config['DROPOUT'],
                       chr_pad_idx)
        
        highway_config = self.config['MODEL']['HIGHWAY']
#         cnn_output_dim = sum([(self.config['DATA']['CHR_MAX_LEN'] - fs + 1) for fs in cnn_config['FILTER_SIZES']])
        cnn_output_dim = len(cnn_config['FILTER_SIZES']) * cnn_config['N_FILTERS']
        print('cnn_output_dim', cnn_output_dim)
        self.highway = Highway(cnn_output_dim, highway_config['HIGHWAY_N_LAYERS'], f=torch.nn.functional.relu)
        
        
        lstm_config = self.config['MODEL']['LSTM']
        self.rnn = LSTM_LM(cnn_output_dim, predict_dim, lstm_config['HID_DIM'], lstm_config['N_LAYERS'], 
                            lstm_config['DROPOUT'], lstm_config['BIDIRECTIONAL'])
        
        self.criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)
        self.PREDICT_DIM = predict_dim
        
        
    def forward(self, input, finetune=False):
        output = self.cnn(input)
        output = self.highway(output)
        if finetune:
            return self.rnn(output, finetune)
        forward_output, backward_output = self.rnn(output, finetune)
        return forward_output, backward_output


    def training_step(self, batch, batch_nb):
        src_chr, trg = batch
        src_chr, trg = src_chr.to(self.device), trg.to(self.device)
        # forward_output[:, :-1, :] -> (batch, seq_len - 1, output_dim) 
        # forward_output[...].reshape(-1, PREDICT_DIM) -> ( batch * (seq_len - 1), output_dim)

        # trg -> (batch_size, seq_len)
        # trg.reshape(-1) -> (batch_size * seq_len)
        forward_output, backward_output = self(src_chr[:, :-1 :])
        forward_loss = self.criterion(forward_output.reshape(-1, self.PREDICT_DIM), trg.reshape(-1))
        backward_loss = self.criterion(backward_output.reshape(-1, self.PREDICT_DIM), trg.reshape(-1))
        loss = forward_loss + backward_loss
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
