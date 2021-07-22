import mlflow
from .utils import *
from .model.bert import *
from .dataset import *
import torch
import dill
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader



class BERT_trainer(pl.LightningModule):
    def __init__(self, config):
        super(BERT_trainer, self).__init__()
        self.config = config
        data_config = config['data']
        dataset = BERT_Dataset(data_config['src'], data_config['vocab'],
                               data_config['max_len'], data_config['nsp_prob'], data_config['mask_ratio'])
        valid_dataset = BERT_Dataset(data_config['src_valid'], data_config['vocab'],
                               data_config['max_len'], data_config['nsp_prob'])
        self.dataloader = DataLoader(dataset, data_config['batch_size'], collate_fn=pad_collate)
        self.valid_dataloader = DataLoader(valid_dataset, data_config['batch_size'], collate_fn=pad_collate)
        print('before special token', dataset.tokenizer.get_vocab_size())
        dataset.tokenizer.add_special_tokens(['[SEP]', '[CLS]', '[MASK]', '[EOD]'])
        self.vocab_size = dataset.tokenizer.get_vocab_size()
        print('after special token', self.vocab_size)
        self.pad_idx = dataset.tokenizer.token_to_id('[PAD]')
        self.bert = BERT(self.config, self.vocab_size, self.pad_idx) # 하드 코딩 인덱스 에러가 남
        # https://keep-steady.tistory.com/37?category=702926 : speical token 에러인것으로 보임 eod추가할것
        
        device = config['train']['device'] 
        self.bert.to(device)
        self.bert.train()
        self.bert.zero_grad()
        self.bert.apply(self.initialize_weights);
        if device == 'cuda':
            self.gpus = 1
        else:
            self.gpus = 0
            

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
    
        
    def train(self):
        checkpoint_callback = ModelCheckpoint(monitor='valid_total_loss')

        # Add your callback to the callbacks list
        trainer = pl.Trainer(callbacks=[checkpoint_callback],
                            max_epochs=self.config['train']['n_epochs'], 
                             progress_bar_refresh_rate=10,
                             gpus=self.gpus)  

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()
        
        # Train the model
        mlflow.end_run() # 이전에 돌아가고 있던거 끄기
        with mlflow.start_run() as run:
            for key, value in self.config.items():
                mlflow.log_param(key, value)
            trainer.fit(self.bert, self.dataloader, self.valid_dataloader)

        
if __name__ == '__main__':
    config_file = '/home/long8v/torch_study/paper/06_BERT/config.yaml'
    config = read_yaml(config_file)
    print('train started..')
    trainer = BERT_trainer(config)
    trainer.train()