import mlflow
from utils import *
from model import *
from dataset import *
from torch8text import *
import torch
from utils import *
from model import *
from dataset import *
from torch8text import *
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader



class ELMoTrainer(pl.LightningModule):
    def __init__(self, corpus, config):
        super(ELMoTrainer, self).__init__()
        
        self.config = config
        self.petition_ds = PetitionDataset(config)
        self.petition_ds = self.petition_ds(corpus)
        self.petition_dl = DataLoader(self.petition_ds, config['DATA']['BATCH_SIZE'], collate_fn=pad_collate)
        
        self.chr_vocab_size = len(self.petition_ds.chr_field.vocab)
        self.chr_pad_idx = self.petition_ds.chr_field.vocab.stoi_dict['<PAD>']
        self.trg_pad_idx = self.petition_ds.token_field.vocab.stoi_dict['<PAD>']
        self.predict_dim = len(self.petition_ds.token_field.vocab) 
        
        self.elmo = ELMo(self.config, self.chr_vocab_size, self.chr_pad_idx, self.trg_pad_idx, self.predict_dim)
        
        device = config['TRAIN']['DEVICE'] 
        self.elmo.to(device)
        self.elmo.train()
        self.elmo.zero_grad()
        self.elmo.apply(self.initialize_weights);
        
        trainer = pl.Trainer(max_epochs=config['TRAIN']['N_EPOCHS'], progress_bar_refresh_rate=10,
                             gpus=1, auto_lr_find = True)
        

#         Auto log all MLflow entities
        mlflow.pytorch.autolog()
        
        # Train the model
        mlflow.end_run() # 이전에 돌아가고 있던거 끄기
        with mlflow.start_run() as run:
            mlflow.log_params(config)
            trainer.fit(self.elmo, self.petition_dl)
        self.save(f'model/elmo_{get_now()}')


    def initialize_weights(self, m):
        if hasattr(m, 'weight'):
            if m.weight is None:
                print(m)  # weight가 None인 것들이 있음 -> crossentropy loss
            elif m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
    
    def save(self, path):
        mkdir(path)
        torch.save(self.elmo.state_dict(), f'{path}/model.pt')
        model_config = \
        {
            'DATA':
            {
                'CHR_VOCAB_SIZE': self.chr_vocab_size,
                'CHR_PAD_IDX': self.chr_pad_idx,
                'TRG_PAD_IDX': self.trg_pad_idx,
                'PREDICT_DIM': self.predict_dim,
                'MAX_CHR_LEN': self.petition_ds.chr_field.max_len
            }
        }
        self.config.update(model_config)
        save_yaml(self.config, f'{path}/model_config.yaml')
        save_yaml(dict(self.petition_ds.token_field.vocab.stoi_dict), f'{path}/token_dict.yaml')
        save_yaml(dict(self.petition_ds.chr_field.vocab.stoi_dict), f'{path}/chr_dict.yaml')
        print(read_yaml(f'{path}/chr_dict.yaml'))
    
    
if __name__ == '__main__':
    with open('../data/petitions_dev.p', 'rb') as f:
        corpus = pickle.load(f)
    config_file = '/home/long8v/torch_study/paper/05_ELMo/config.yaml'
    config = read_yaml(config_file)
    print('trainer loading..')
    trainer = ELMoTrainer(corpus, config)
    print('start train..')
    trainer.train()