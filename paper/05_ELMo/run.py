import sys
sys.path.append('source/')
sys.path.append('source/finetune/')
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
from trainer_finetune import *


if __name__ == '__main__':
    with open('/home/long8v/torch_study/data/ynat/train_tokenized.ynat', 'r') as f:
        corpus = f.readlines()
    corpus = [(txt.split('\t')[1], txt.split('\t')[0]) for txt in corpus]
    config_file = '/home/long8v/torch_study/paper/05_ELMo/config_finetune.yaml'
    finetune_config = read_yaml(config_file)
    print(finetune_config)
    with open('/home/long8v/torch_study/data/ynat/val_tokenized.ynat', 'r') as f:
        corpus_valid = f.readlines()
    corpus_valid = [(txt.split('\t')[1], txt.split('\t')[0]) for txt in corpus_valid]
    config_file = '/home/long8v/torch_study/paper/05_ELMo/config_finetune.yaml'
    finetune_config = read_yaml(config_file)
    print('trainer loading..')
    trainer = gruTrainer(corpus, corpus_valid, finetune_config)
    trainer.train()