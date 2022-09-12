import sys

sys.path.append("source/")
sys.path.append("source/finetune/")
import mlflow
from utils import *
from model import *
from model import *
from dataset import *
from torch8text import *
from trainer import *
import torch
from utils import *
from model import *
from dataset import *
from torch8text import *
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# from torch.utils.data import Dataset, DataLoaders
from trainer_finetune import *


if __name__ == "__main__":
    with open("./data/petitions.p", "rb") as f:
        corpus = pickle.load(f)
    config_file = "~/torch_study/paper/05_ELMo/config.yaml"
    config = read_yaml(config_file)
    print("trainer loading..")
    trainer = ELMoTrainer(corpus, config)
    print("start train..")
    trainer.train()
