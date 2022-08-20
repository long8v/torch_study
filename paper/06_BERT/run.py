import mlflow
from source.utils import *
from source.trainer import *
from source.model.bert import *
from source.dataset import *
from omegaconf import Omegaconf
import torch
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader


if __name__ == "__main__":
    config_file = "~/torch_study/paper/06_BERT/config.yaml"
    # config = read_yaml(config_file)
    config = Omegaconf(config_file)
    print("trainer loading..")
    trainer = BERT_trainer(config)
    print("start train..")
    trainer.train()
