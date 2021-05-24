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
    def __init__(self, dataset, config):
        super(gruTrainer, self).__init__()
        
        self.config = config
        self.petition_ds = PetitionDataset_finetune(config)
        self.petition_ds = self.petition_ds(dataset)
        self.petition_dl = DataLoader(self.petition_ds, batch_size=4, collate_fn=pad_collate_finetune)
        INPUT_DIM = len(self.petition_ds.chr_field.vocab.stoi_dict)
        OUTPUT_DIM = len(self.petition_ds.label_field.vocab.stoi_dict)
        N_LAYERS = 2
        HID_DIM = 512
        EMBEDDING_DIM = 1024
        self.simple_gru = simpleGRU_model(self.config, INPUT_DIM, EMBEDDING_DIM, N_LAYERS, HID_DIM, OUTPUT_DIM)
        device = config['TRAIN']['DEVICE'] 
        self.simple_gru.to(device)
        self.simple_gru.train()
        self.simple_gru.zero_grad()
        self.simple_gru.apply(self.initialize_weights);
        
        trainer = pl.Trainer(max_epochs=self.config['TRAIN']['N_EPOCHS'], progress_bar_refresh_rate=1, gpus=1)

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()
        

        
        # Train the model
        mlflow.end_run() # 이전에 돌아가고 있던거 끄기
        with mlflow.start_run() as run:
            mlflow.log_params(config)
            trainer.fit(self.simple_gru, self.petition_dl)

        # fetch the auto logged parameters and metrics
#         print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

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
    config_file = '/home/long8v/torch_study/paper/05_ELMo/config_finetune.yaml'
    config = read_yaml(config_file)
    print('trainer loading..')
    trainer = gruTrainer(corpus, config)
    print('start train..')
    trainer.train()