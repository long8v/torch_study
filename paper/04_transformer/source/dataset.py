import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Multi30k_Dataset:
    def __init__(self):
        super().__init__()
        print('loading_tokenizers..')
        self.spacy_fr = spacy.load('fr_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        self.SRC = Field(tokenize = self.tokenize_fr, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True,
            batch_first = True)

        self.TRG = Field(tokenize = self.tokenize_en, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True, 
                    batch_first = True)
        
    def __call__(self, batch_size):
        print('loading datset..')
        train_data, valid_data, test_data = Multi30k.splits(exts = ('.fr', '.en'), 
                                                    fields = (self.SRC, self.TRG),
                                                   root = '/home/long8v/torch_study/paper/03_attention/.data/')
        print('build vocabs..')
        self.SRC.build_vocab(train_data, min_freq = 2)
        self.TRG.build_vocab(train_data, min_freq = 2)
        train_iter, valid_iter, test_iter = BucketIterator.splits((train_data, valid_data, test_data), batch_size = batch_size)
        return train_iter, valid_iter, test_iter
    
    def tokenize_fr(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_fr.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
    

if __name__ == '__main__':
    multi30k = Multi30k_Dataset()
    train_iter, _, _ = multi30k(batch_size=32)
    for i in train_iter:
        print(i.src)
        print(i.trg)