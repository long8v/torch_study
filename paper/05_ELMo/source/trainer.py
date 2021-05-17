from utils import *
from model import *
from dataset import *
from torch8text import *
import torch
from utils import *
from model import *
from dataset import *
from torch8text import *


class ELMoTrainer:
    def __init__(self, dataset, config):
        self.config = config
        self.petition_ds = PetitionDataset(config)
        self.petition_dl = self.petition_ds(corpus)
        chr_vocab_size = len(self.petition_ds.chr_field.vocab)
        chr_pad_idx = self.petition_ds.chr_field.vocab.stoi_dict['<PAD>']
        trg_pad_idx = self.petition_ds.mecab_field.vocab.stoi_dict['<PAD>']
        cnn_config = self.config['MODEL']['CNN']
        self.cnn = CNN(chr_vocab_size, cnn_config['EMBEDDING_DIM'],
                       cnn_config['N_FILTERS'], cnn_config['FILTER_SIZES'], 
                       cnn_config['OUTPUT_DIM'], cnn_config['DROPOUT'],
                       chr_pad_idx)
        
        highway_config = self.config['MODEL']['Highway']
        cnn_output_dim = sum(cnn_config['FILTER_SIZES'])
        self.highway = Highway(cnn_output_dim, highway_config['HIGHWAY_N_LAYERS'], f=torch.nn.functional.relu)
        self.PREDICT_DIM = len(self.petition_ds.mecab_field.vocab) 
        
        lstm_config = self.config['MODEL']['LSTM']
        self.lstm = LSTM_LM(cnn_output_dim, self.PREDICT_DIM, lstm_config['HID_DIM'], lstm_config['N_LAYERS'], 
                            lstm_config['DROPOUT'], lstm_config['BIDIRECTIONAL'])
        
        self.elmo = ELMo(self.cnn, self.highway, self.lstm)
        self.device = config['TRAIN']['DEVICE'] 
        self.elmo.to(self.device)
        self.optimizer = optim.Adam(self.elmo.parameters(), lr = config['TRAIN']['LR'])
        self.criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)
        self.elmo.apply(self.initialize_weights);
    
    def train(self):
        log_epoch_loss = []
        n_pass = 0
        for epoch in range(self.config['TRAIN']['N_EPOCHS']):
            print(f'Epoch {epoch}')
            self.optimizer.zero_grad()
            i = 0 
            epoch_loss = 0
            for src_chr, trg in self.petition_dl:
                src_chr, trg = src_chr.to(self.device), trg.to(self.device)
                forward_output, backward_output = self.elmo(src_chr)
                forward_loss = self.criterion(forward_output[:, :-1, :].reshape(-1, self.PREDICT_DIM), trg.reshape(-1))
                backward_loss = self.criterion(backward_output[:, :-1, :].reshape(-1, self.PREDICT_DIM), trg.reshape(-1))
                loss = forward_loss + backward_loss
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                i += 1
            log_epoch_loss += [epoch_loss / i]
        torch.save(self.elmo.state_dict(), f'model_{int(log_epoch_loss[-1])}.pt')
        return log_epoch_loss
    
    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
        
    def evaluate(self):
        pass
    
    def predict(self):
        pass
    

    
    
if __name__ == '__main__':
    with open('../data/petitions.p', 'rb') as f:
        corpus = pickle.load(f)
    config = read_yaml('../config.yaml')
    print('trainer loading..')
    trainer = ELMoTrainer(corpus, config)
    print('start train..')
    trainer.train()