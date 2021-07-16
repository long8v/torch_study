import mlflow
import sys
sys.path.append('/home/long8v/torch_study/paper/06_BERT/source')
from utils import *
from finetune.ner_bert import *
from finetune.dataset import *
from finetune.trainer import *

if __name__ == '__main__':
    config_file = '/home/long8v/torch_study/paper/06_BERT/config_finetune.yaml'
    config = read_yaml(config_file)
    print('train started..')
    trainer = NER_BERT_trainer(config)