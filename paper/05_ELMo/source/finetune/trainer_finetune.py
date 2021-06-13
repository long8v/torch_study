import sys
sys.path.append('..')
import mlflow
from utils import *
from model import *
from model_finetune import *
from dataset import *
from torch8text import *
import torch
from utils import *
from model import *
from dataset_finetune import *
from torch8text import *
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader



class gruTrainer(pl.LightningModule):
    def __init__(self, dataset, dataset_valid, finetune_config):
        super(gruTrainer, self).__init__()
        self.finetune_config = finetune_config
                       
        elmo_path = self.finetune_config['ELMO']['PATH']
        pt_file = torch.load(f'{elmo_path}/model.pt')
    
        elmo_config = read_yaml(f'{elmo_path}/model_config.yaml')
        
        data_config = elmo_config['DATA']
        chr_vocab_size = data_config['CHR_VOCAB_SIZE']
        chr_pad_idx = data_config['CHR_PAD_IDX']
        trg_pad_idx = data_config['TRG_PAD_IDX']
        predict_dim = data_config['PREDICT_DIM']
        MAX_CHR_LEN = data_config['MAX_CHR_LEN']
        
        elmo = ELMo(elmo_config, chr_vocab_size, chr_pad_idx, trg_pad_idx, predict_dim)
        elmo.load_state_dict(pt_file)
        
        token_stoi_dict = read_yaml(f'{elmo_path}/token_dict.yaml')
        chr_stoi_dict = read_yaml(f'{elmo_path}/chr_dict.yaml')
        
        print(elmo_config)
        self.petition_ds = PetitionDataset_finetune(finetune_config)
        self.petition_ds = self.petition_ds(dataset, token_stoi_dict, chr_stoi_dict)
        self.petition_dl = DataLoader(self.petition_ds, batch_size=self.finetune_config['DATA']['BATCH_SIZE'], collate_fn=pad_collate_finetune)
        
        if dataset_valid:
            self.petition_ds = PetitionDataset_finetune(finetune_config)
            self.petition_ds_valid = self.petition_ds(dataset_valid, token_stoi_dict, chr_stoi_dict)
            self.petition_dl_valid = DataLoader(self.petition_ds_valid, batch_size=self.finetune_config['DATA']['BATCH_SIZE'], collate_fn=pad_collate_finetune)

        INPUT_DIM = elmo_config['DATA']['PREDICT_DIM'] # now we have one input as token
        OUTPUT_DIM = len(self.petition_ds.label_field.vocab.stoi_dict)
        
        model_config = self.finetune_config['MODEL']['GRU']
        N_LAYERS = model_config['N_LAYERS']
        HID_DIM = model_config['HID_DIM']
        EMBEDDING_DIM = model_config['EMBEDDING_DIM']
        
        USE_ELMO = self.finetune_config['MODEL']['ELMO']
        if USE_ELMO:
            self.simple_gru = simpleGRU_model_w_elmo(self.finetune_config, elmo, INPUT_DIM, EMBEDDING_DIM, MAX_CHR_LEN,
                                          N_LAYERS, HID_DIM, OUTPUT_DIM)
        else:
            self.simple_gru = simpleGRU_model(self.finetune_config, INPUT_DIM, EMBEDDING_DIM, 
                                          N_LAYERS, HID_DIM, OUTPUT_DIM)
        
        device = finetune_config['TRAIN']['DEVICE'] 
        self.simple_gru.to(device)
        self.simple_gru.train()
        self.simple_gru.zero_grad()
        self.simple_gru.apply(self.initialize_weights);
        
        trainer = pl.Trainer(max_epochs=self.finetune_config['TRAIN']['N_EPOCHS'], 
                             progress_bar_refresh_rate=10, gpus=1,
                             auto_lr_find=True)

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()
        
        # Train the model
        mlflow.end_run() # 이전에 돌아가고 있던거 끄기
        with mlflow.start_run() as run:
            mlflow.log_params(self.finetune_config)
            trainer.fit(self.simple_gru, self.petition_dl, self.petition_dl_valid)
            

    def initialize_weights(self, m):
        if hasattr(m, 'weight'):
            if m.weight is None:
                print(m)
            elif m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)  

    
    
if __name__ == '__main__':
    with open('/home/long8v/torch_study/data/ynat/train_tokenized.ynat', 'r') as f:
        corpus = f.readlines()
    corpus = [(txt.split('\t')[1], txt.split('\t')[0]) for txt in corpus]
    corpus = corpus[:1024]
    with open('/home/long8v/torch_study/data/ynat/val_tokenized.ynat', 'r') as f:
        corpus_valid = f.readlines()
    corpus_valid = [(txt.split('\t')[1], txt.split('\t')[0]) for txt in corpus_valid]
    corpus_valid = corpus_valid[:1024]
    config_file = '/home/long8v/torch_study/paper/05_ELMo/config_finetune.yaml'
    finetune_config = read_yaml(config_file)
    print('trainer loading..')
    trainer = gruTrainer(corpus, corpus_valid, finetune_config)