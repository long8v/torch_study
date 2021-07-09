# https://keep-steady.tistory.com/37#recentEntries
# https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer

from torch.utils.data import Dataset, DataLoader
import random
import numpy.random
from collections import namedtuple  
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import math
import linecache
from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
import sys
sys.path.append('/home/long8v/torch_study/paper/05_ELMo/source/')
from torch8text.data import *

class NER_Dataset(Dataset):
    def __init__(self, corpus_path, tokenizer_path,
                 unk_token='[UNK]', categories = ['QT', 'LC', 'PS', 'OG', 'DT', 'TI']):
        self.corpus_path = corpus_path
        with open(self.corpus_path, 'r') as f:
            corpus = f.read()
        corpus = corpus.split('\n\n')
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        splitted_corpus = [[sen.split('\t') 
                                for sen in sentence.split('\n')] 
                                for sentence in corpus 
                                if '\t' in sentence]
        splitted_corpus = [[sen for sen in sentence 
                                 if len(sen) > 1] 
                                for sentence in splitted_corpus]
        corpus_pair = [[[char, bio] 
                             for char, bio in corpus 
                             if bio[:2] in ['B-', 'I-', 'O']] 
                            for corpus in splitted_corpus]
#         print(corpus_bio)
        self.corpus_char = [''.join([char for char, bio in corpus]) 
                       for corpus in corpus_pair]
        self.corpus_bio = [[bio for char, bio in corpus] 
                      for corpus in corpus_pair]
#         print(self.corpus_bio)
        self.unk_token = unk_token
        self.all_labels = ['O'] + [f'{bio}-{cat}' 
                                   for bio in ['B', 'I']
                                   for cat in categories]
        self.label_field = LabelField()
        self.label_field.build_vocab(self.all_labels)
        print(self.label_field.vocab.itos_dict)
        
    def __len__(self):
        return len(self.corpus_char)
    
    def __getitem__(self, ids):
        text, label = self.corpus_char[ids], self.corpus_bio[ids]
        label = self.get_token_labels(text, label, self.tokenizer)
        text = self.tokenizer.encode(text).ids
        label = self.label_field.process(label)
        return torch.tensor(text).long(), torch.tensor(label).long()
    
    def get_token_labels(self, text, label, tokenizer):
        tokenized = tokenizer.encode(text)
        token_word = tokenized.tokens
        offset = tokenized.offsets
        index = 0
        token_labels = []
        label_clean = [lbl for txt, lbl in list(zip(text, label))
                       if txt.strip()]
        try:
            for token_off, token in zip(offset, token_word):
                len_token_clean = token_off[1] - token_off[0] 
                token_labels.append(label_clean[index:index+len_token_clean][0]) # 가장 첫번째 bio 태그를 태그로 사용
                index += len_token_clean
        except:
            print(token_word, label_clean)
        return token_labels

def pad_collate(batch):
    text, label = zip(*batch)
    named_tuple = namedtuple('data', ['text', 'label'])
    text_pad = pad_sequence(text, batch_first=True, padding_value=0) # 하드코딩 고쳐야함
    label_pad = pad_sequence(label, batch_first=True, padding_value=0)
    return named_tuple(text_pad, label_pad)   
        

if __name__ == '__main__':
    ds = NER_Dataset('/home/long8v/torch_study/paper/file/klue-ner-v1_train.tsv',
                       '/home/long8v/torch_study/paper/file/bert/vocab.json',
                    '[UNK]')
    print(ds[0])
    
    dl = DataLoader(ds, batch_size=16, collate_fn=pad_collate)
    for _ in dl:
        print(_.text.shape)
        print(_.label.shape)