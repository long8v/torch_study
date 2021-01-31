import os
import os.path as osp
import random
import re
import pickle
import argparse
import numpy as np
from random import randint
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from gensim.models import KeyedVectors

class Vocab:    
    def build_vocabs(self, sentence_list):
        from collections import defaultdict
        self.stoi_dict = defaultdict(lambda: 0) # 원래 <UNK>로 되어있었음
        self.stoi_dict['<UNK>'] = 0
        self.stoi_dict['<PAD>'] = 1
        _index = 2
        for sentence in sentence_list:
            tokens_list = sentence
            for word in tokens_list:
                if word in self.stoi_dict:
                    pass
                else:
                    self.stoi_dict[word] = _index
                    _index += 1
        self.itos_dict = {v:k for k, v in self.stoi_dict.items()}
        
    def stoi(self, token_list):
        return [self.stoi_dict[word] for word in token_list]

    def itos(self, indices):
        return " ".join([self.itos_dict[index] for index in indices if self.itos_dict[index] != '<PAD>'])


class CNNDataset: # 굳이 Dataset 상속을 안해줘도 된다고 함
    def __init__(self, path, w2v):
        zipped_data = list(zip(*data))
        
        # 전처리하는 과정 __getitem__에서 안 한 이유는 vocab 만들때 같은 전처리를 사용해야해서..!!
        self.text = zipped_data[0]
        self.text = [self.clean_str(sen) for sen in self.text]
        self.text = [[word for word in self.tokenizer(sen)] for sen in self.text]
        self.label = zipped_data[1]
        
        # vocab 만들기 -> class 안에 다른 class instance를 정의하는게 보편적인지는 잘 모르겠음
        # ...이렇게 하면 문제점이 생기는게, train, valid, test 따로따로 build_vocab을 만들어서 안됨!!! 어떡하지
        self.vocab = Vocab()
        self.vocab.build_vocabs(self.text)    
        self.pretrained_embedding = self.get_pretrained_embeddings()
        self.w2v = w2v

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        sample_label = self.label[idx]
        sample_text = self.text[idx]
        sample_text = self.vocab.stoi(sample_text)
        return torch.Tensor(sample_text).long(), sample_label
    
    def tokenizer(self, sentence):
        return sentence.split()
    
    def get_pretrained_embeddings(self):
        pretrained_embedding = []
        for word in self.vocab.stoi_dict:
            if word in w2v:
                pretrained_embedding.append(w2v[word])
            else: 
                pretrained_embedding.append(np.random.uniform(-0.25, 0.25, 300))
        return torch.from_numpy(np.array(pretrained_embedding))        
    
    def clean_str(self, string, TREC=False):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string)     
        return string.strip() if TREC else string.strip().lower()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('-p', '--path', type=str, default='../data')
    parser.add_argument('-p', '--path', type=str, default='../data')
    args = parser.parse_args()

    dataset = MR(args)
    data_loader = DataLoader(dataset=dataset, 
                            batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn)

    for i in data_loader:
        print(i)
        break 