import mlflow
from .utils import *
from .model.bert import *
from .dataset import *
import torch
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader



class BERT_trainer(pl.LightningModule):
    def __init__(self, config):
        super(BERT_trainer, self).__init__()
        self.config = config
        data_config = config['data']
        dataset = BERT_Dataset(data_config['src'], data_config['vocab'],
                               data_config['max_len'], data_config['nsp_prob'])
        valid_dataset = BERT_Dataset(data_config['src_valid'], data_config['vocab'],
                               data_config['max_len'], data_config['nsp_prob'])
        dataloader = DataLoader(dataset, data_config['batch_size'], collate_fn=pad_collate)
        valid_dataloader = DataLoader(valid_dataset, data_config['batch_size'], collate_fn=pad_collate)
        vocab_size = dataset.tokenizer.get_vocab_size()
        pad_idx = dataset.tokenizer.token_to_id('[PAD]')
        self.bert = BERT(self.config, vocab_size + 10, pad_idx)
        
        device = config['train']['device'] 
        self.bert.to(device)
        self.bert.train()
        self.bert.zero_grad()
        self.bert.apply(self.initialize_weights);
        if device == 'cuda':
            gpus = 1
        else:
            gpus = 0
            
        trainer = pl.Trainer(max_epochs=config['train']['n_epochs'], progress_bar_refresh_rate=10,
                             gpus=gpus, auto_lr_find = True)
        

#         Auto log all MLflow entities
        mlflow.pytorch.autolog()
        
        # Train the model
        mlflow.end_run() # 이전에 돌아가고 있던거 끄기
        with mlflow.start_run() as run:
            mlflow.log_params(config)
            trainer.fit(self.bert, dataloader, valid_dataloader)
        self.save(f'model/ebert_{get_now()}')


    def initialize_weights(self, m):
        if hasattr(m, 'weight'):
            if m.weight is None:
                print(m)  # weight가 None인 것들이 있음 -> crossentropy loss
            elif m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
    
    def save(self, path):
        mkdir(path)
        torch.save(self.bert.state_dict(), f'{path}/model.pt')
    
if __name__ == '__main__':
    config_file = '/home/long8v/torch_study/paper/06_BERT/config.yaml'
    config = read_yaml(config_file)
    print('trainer loading..')
    trainer = BERT_trainer(config)
    print('start train..')
    trainer.train()