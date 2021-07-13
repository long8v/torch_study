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
    def __init__(self, corpus_path, tokenizer_path, categories = ['QT', 'LC', 'PS', 'OG', 'DT', 'TI']):
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
        self.corpus_char = [''.join([char for char, bio in corpus]) 
                       for corpus in corpus_pair]
        self.corpus_bio = [[bio for char, bio in corpus] 
                      for corpus in corpus_pair]
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
                token_labels.append(label_clean[index:index+len_token_clean][0]) 
                # 가장 첫번째 bio 태그를 태그로 사용
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
                       '/home/long8v/torch_study/paper/file/bert/vocab.json')
    def decode_from_tensor(ids):
        print(ds.tokenizer.decode(ids.tolist(), skip_special_tokens=False))
    decode_from_tensor(ds[0][0])
    print(ds.label_field.vocab.itos(ds[0][1].tolist()))
    print(ds[0][0])
    print(ds[0][1])
    
    
    dl = DataLoader(ds, batch_size=16, collate_fn=pad_collate)
    for _ in dl:
        print(_.text.shape)
        print(_.label.shape)
        break
        
# {0: 'I-OG', 1: 'I-DT', 2: 'I-QT', 3: 'I-TI', 4: 'O', 5: 'B-LC', 6: 'B-DT', 7: 'B-PS', 8: 'I-PS', 9: 'B-TI', 10: 'B-QT', 11: 'B-OG', 12: 'I-LC'}
# 특히 영동고속도로 [UNK] 방향 문막휴게소에서 만종분기점까지 [UNK] 구간에는 승용차 전용 임시 [UNK] 운영하기로 했다.
# O B-LC I-LC B-LC O B-LC I-LC I-LC I-LC I-LC O O B-LC I-LC I-LC I-LC I-LC O O B-QT O O O O O O O O O O O O O O O O
# tensor([2195, 8395, 8105,    1, 2627,  426, 1261, 1458, 1414, 1134, 1102, 1010,
#          383, 1123, 1011, 1023, 1108, 1197, 1128,    1, 2910, 1102, 1375,  573,
#         1191, 1255, 2983, 3786,    1, 2003, 1053, 1023, 1092,  952, 1025,    8])
# tensor([ 4,  5, 12,  5,  4,  5, 12, 12, 12, 12,  4,  4,  5, 12, 12, 12, 12,  4,
#          4, 10,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4])
# torch.Size([16, 112])
# torch.Size([16, 112])