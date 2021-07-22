import mlflow
import sys
sys.path.append('/home/long8v/torch_study/paper/06_BERT/source')
from utils import *
from finetune.ner_bert import *
from finetune.dataset import *
import torch
import dill
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader



class NER_BERT_trainer(pl.LightningModule):
    def __init__(self, config):
        super(NER_BERT_trainer, self).__init__()
        self.config = config
        data_config = config['data']
        dataset = NER_Dataset(data_config['src'], data_config['vocab'])
        valid_dataset = NER_Dataset(data_config['src_valid'], data_config['vocab'])
        dataloader = DataLoader(dataset, data_config['batch_size'], collate_fn=pad_collate)
        valid_dataloader = DataLoader(valid_dataset, data_config['batch_size'], collate_fn=pad_collate)
        print('before special token', dataset.tokenizer.get_vocab_size())
        dataset.tokenizer.add_special_tokens(['[SEP]', '[CLS]', '[MASK]', '[EOD]'])
        self.vocab_size = dataset.tokenizer.get_vocab_size()
        print('after special token', self.vocab_size)
        self.pad_idx = dataset.tokenizer.token_to_id('[PAD]')
        self.output_dim = dataset.output_dim
        print(self.output_dim)
        self.ner_bert = NER_BERT(self.config, self.vocab_size, self.output_dim, self.pad_idx) 
        
        device = config['train']['device'] 
        self.ner_bert.to(device)
        self.ner_bert.train()
        self.ner_bert.zero_grad()
        self.ner_bert.fcn.apply(self.initialize_weights);
        if device == 'cuda':
            gpus = 1
        else:
            gpus = 0
            
        trainer = pl.Trainer(max_epochs=config['train']['n_epochs'], 
                             progress_bar_refresh_rate=10,
                             gpus=gpus, auto_lr_find= True)
        

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()
        
        # Train the model
        mlflow.end_run() # 이전에 돌아가고 있던거 끄기
        with mlflow.start_run() as run:
            for key, value in config.items():
                mlflow.log_param(key, value)
            trainer.fit(self.ner_bert, dataloader, valid_dataloader)
        self.save(f'model/ner_bert_{get_now()}')

# https://github.com/GyuminJack/torchstudy/blob/main/06Jun/BERT/src/trainer.py
    def initialize_weights(self, m):
        for name, param in m.named_parameters():
            if ("fc" in name) or ('embedding' in name):
                if 'bias' in name:
                    torch.nn.init.zeros_(param.data)
                else:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
            elif "layer_norm" in name:
                if 'bias' in name:
                    torch.nn.init.zeros_(param.data)
                else:
                    torch.nn.init.constant_(param.data, 1.0)
    
    def save(self, path):
        mkdir(path)
        torch.save(self.bert.state_dict(), f'{path}/model.pt')

if __name__ == '__main__':
    config_file = '/home/long8v/torch_study/paper/06_BERT/config_finetune.yaml'
    config = read_yaml(config_file)
    print('train started..')
    trainer = NER_BERT_trainer(config)