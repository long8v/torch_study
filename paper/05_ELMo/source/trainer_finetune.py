import mlflow
from utils import *
from model_finetune import *
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



class gruTrainer(pl.LightningModule):
    def __init__(self, dataset, elmo_config, finetune_config):
        super(gruTrainer, self).__init__()
        self.elmo_config = elmo_config
        self.finetune_config = finetune_config
        
        
        ### 하드 코딩 : 모델 어딘가에 저장되도록 해야함
        chr_vocab_size = 1299 
        chr_pad_idx = 0
        trg_pad_idx = 0
        predict_dim = 18216
                
        elmo = ELMo(elmo_config, chr_vocab_size, chr_pad_idx, trg_pad_idx, predict_dim)
        PATH = self.finetune_config['ELMO']['PATH']
        checkpoint = torch.load(PATH)
        elmo.load_state_dict(checkpoint['state_dict'])
        
        self.petition_ds = PetitionDataset_finetune(elmo_config)
        self.petition_ds = self.petition_ds(dataset)
        self.petition_dl = DataLoader(self.petition_ds, batch_size=4, collate_fn=pad_collate_finetune)
        
        INPUT_DIM = len(self.petition_ds.chr_field.vocab.stoi_dict)
        OUTPUT_DIM = len(self.petition_ds.label_field.vocab.stoi_dict)
        
        model_config = finetune_config['MODEL']['GRU']
        N_LAYERS = model_config['N_LAYERS']
        HID_DIM = model_config['HID_DIM']
        EMBEDDING_DIM = model_config['EMBEDDING_DIM']
        
        self.simple_gru = simpleGRU_model_w_elmo(self.finetune_config, elmo, INPUT_DIM, EMBEDDING_DIM, 
                                          N_LAYERS, HID_DIM, OUTPUT_DIM)
        
        device = finetune_config['TRAIN']['DEVICE'] 
        self.simple_gru.to(device)
        self.simple_gru.train()
        self.simple_gru.zero_grad()
        self.simple_gru.apply(self.initialize_weights);
        
        trainer = pl.Trainer(max_epochs=self.finetune_config['TRAIN']['N_EPOCHS'], 
                             progress_bar_refresh_rate=10, gpus=1)

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()
        

        
        # Train the model
        mlflow.end_run() # 이전에 돌아가고 있던거 끄기
        with mlflow.start_run() as run:
            mlflow.log_params(self.finetune_config)
            trainer.fit(self.simple_gru, self.petition_dl)

    def initialize_weights(self, m):
        if hasattr(m, 'weight'):
            if m.weight is None:
                print('?? why none') # weight가 None인 것들이 있음
            elif m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
        
    def evaluate(self):
        pass
    
    def predict(self):
        pass
    

    
    
if __name__ == '__main__':
    with open('../data/petitions_2019-01.txt', 'r') as f:
        corpus = f.readlines()
    json_list = [eval(json.strip()) for json in corpus]
    corpus = [(json['content'], json['category']) for json in json_list]
    corpus = corpus[:1037] 
    config_file = '/home/long8v/torch_study/paper/05_ELMo/config.yaml'
    elmo_config = read_yaml(config_file)
    config_file = '/home/long8v/torch_study/paper/05_ELMo/config_finetune.yaml'
    finetune_config = read_yaml(config_file)
    print('trainer loading..')
    trainer = gruTrainer(corpus, elmo_config, finetune_config)
    print('start train..')
    trainer.train()