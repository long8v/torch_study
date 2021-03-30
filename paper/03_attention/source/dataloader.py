import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

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

class Multi30k_dataset:
    def __init__(self, src = 'en', trg = 'fr', min_freq = 2):
        self.min_freq = min_freq
        self.src = src
        self.trg = trg
        self.src_tokenizer = self.spacy_load_tokenizer(self.src)
        self.trg_tokenizer = self.spacy_load_tokenizer(self.trg)
        print(self.src_tokenizer, self.trg_tokenizer)
        self.src_tokenizer = lambda text : self.spacy_tokenize(text, self.src_tokenizer)
        self.trg_tokenizer = lambda text : self.spacy_tokenize(text, self.trg_tokenizer)
        self.src_field = self.get_field(self.src_tokenizer)
        self.trg_field = self.get_field(self.trg_tokenizer)
        print('load multi30k')
        self.train_data, self.valid_data, self.test_data = self.load_Multi30k()
        print('field_vocab')
        self.field_build_vocab()

    def spacy_load_tokenizer(self, lan):
        try:
            tokenizer = spacy.load(f'{lan}_core_news_sm')
        except:
            tokenizer = spacy.load(f'{lan}_core_web_sm')
        return tokenizer

    def spacy_tokenize(self, text, tokenizer):
        return [token.text 
            for token in tokenizer.tokenizer(text)]

    def get_field(self, tokenizer):
        return Field(tokenize = tokenizer, 
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = True)

    def load_Multi30k(self):
        # this takes so long.. 
        return Multi30k.splits(exts = (f'.{self.src}', f'.{self.trg}'), 
                                        fields = (self.src_field, self.trg_field))

    def field_build_vocab(self):
        print('build vocab for source')
        self.src_field.build_vocab(self.train_data, min_freq = self.min_freq)
        print('build vocab for target')
        self.trg_field.build_vocab(self.train_data, min_freq = self.min_freq)


class Multi30k_iterator:
    def __init__(self, multi30k_dataset, BATCH_SIZE):
        self.train_iter, self.valid_iter, self.test_iter = BucketIterator.splits(
            (multi30k_dataset.train_data, 
            multi30k_dataset.valid_data, 
            multi30k_dataset.test_data), 
            batch_size = BATCH_SIZE)



if __name__ == '__main__':
    ds = Multi30k_dataset()
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-p', '--path', type=str, default='/home/long8v')
    parser.add_argument('-w', '--w2v_path', type=str, default='/home/long8v/Downloads/GoogleNews-vectors-negative300.bin.gz')
    args = parser.parse_args()