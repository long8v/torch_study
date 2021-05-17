import mecab
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import namedtuple  
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from utils import *
import sys
sys.path.append('../source')
from txt_cleaner.clean.master import MasterCleaner
from txt_cleaner.utils import *
from torch8text.data import Vocab, Field

import pickle

class ELMoDataset(Dataset):
    def __init__(self, src, mecab_field, chr_field, max_len):
        self.src = src
        self.mecab_field = mecab_field
        self.chr_field = chr_field
        self.named_tuple = namedtuple('data', ['src_chr', 'trg'])
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.named_tuple(self.getitem(idx, is_char=True), self.getitem(idx)[1:]) 
    
    def getitem(self, idx, is_char=False):
        data = self.src[idx]
        tokenize_data = self.mecab_field.preprocess(data)[:self.max_len]
        if is_char:
            chrs = self.chr_field.preprocess(tokenize_data)
            pad_chrs = self.chr_field.pad_process(tokenize_data, max_len = self.max_len)
            return pad_chrs
        return torch.Tensor(self.mecab_field.vocab.stoi(tokenize_data)).long()
    

class ELMoDataLoader:
    def __init__(self, batch_size = 64):
        self.batch_size = batch_size 
    
    def __call__(self, dataset):
        return DataLoader(dataset, batch_size = self.batch_size, collate_fn = self.pad_collate)
    
    def pad_collate(self, batch):
        (src_chr, trg) = zip(*batch)
        named_tuple = namedtuple('data', ['src_chr', 'trg'])
        src_chr_pad = pad_sequence(src_chr, batch_first=True, padding_value=0)
        trg_pad = pad_sequence(trg, batch_first=True, padding_value=0)
        return named_tuple(src_chr_pad, trg_pad)
    
class PetitionDataset:
    def __init__(self, config):
        self.config = config['DATA']
        print(self.config)
        self.mecab_tokenizer = mecab.MeCab()
        self.token_cleaner = MasterCleaner({'minimum_space_count': self.config['minimum_space_count']})
        self.chr_cleaner = MasterCleaner({'minimum_space_count':0})
        self.mecab_field = Field(tokenize = self.tokenize_pos, 
                                preprocessing = lambda e: self.token_cleaner.cleaning(e),
                                init_token = False,
                                eos_token = False,
                                max_len = self.config['token_max_len']
                                )
        self.chr_field = Field(tokenize = list, 
                                preprocessing = lambda e: self.chr_cleaner.cleaning(e) if len(e) > 1 else e,
                                init_token = False,
                                eos_token = False,
                                max_len = self.config['chr_max_len']
                                )
        self.dataloader = ELMoDataLoader(batch_size = self.config['batch_size'])
        
    def __call__(self, corpus):
        sent_corpus = [sent
                        for text in corpus 
                        for sent in sent_tokenize(text)
                        if self.token_cleaner.cleaning(sent)]
        
        self.mecab_field.build_vocab(sent_corpus)
        self.chr_field.build_vocab(sent_corpus)
        ds = ELMoDataset(sent_corpus, self.mecab_field, self.chr_field, max_len = self.chr_field.max_len)
        dl = self.dataloader(ds)
        return dl
    
    def tokenize_pos(self, inp):
        if type(inp) == str:
            return self.mecab_tokenizer.morphs(inp)
        if type(inp) == list:
            return [self.tokenize_pos(i) for i in inp]

        
        
    
if __name__ == '__main__':
    with open('../data/petitions.p', 'rb') as f:
        corpus = pickle.load(f)
    config = read_yaml('../config.yaml')
    pet_ds = PetitionDataset(config)
    dl = pet_ds(corpus)
    for _ in dl:
        print(_.src_chr)
        print(_.trg)
        break