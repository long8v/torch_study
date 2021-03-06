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

class BERT_Dataset(Dataset):
    def __init__(self, corpus_path, tokenizer_path, max_len,
                 nsp_ratio, mask_ratio=0.15,
                mask_token='[MASK]'):
        self.corpus_path = corpus_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.max_len = max_len     
        self.num_lines = sum(1 for line in open(corpus_path))
        self.nsp_ratio = nsp_ratio
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
    
    def __len__(self):
        return self.num_lines - 1 # because it is pair
    
    def __getitem__(self, ids):
        tokenizer = self.tokenizer
        _sep, _cls, _mask = tokenizer.token_to_id('[SEP]'), tokenizer.token_to_id('[CLS]'), tokenizer.token_to_id(self.mask_token)        
        ids = ids + 1
        senA = linecache.getline(self.corpus_path, ids)
        if senA.strip() == '[EOD]':
            while senA.strip() != '[EOD]':
                ids = random.randint(1, self.num_lines)
                senA = linecache.getline(self.corpus_path, ids)
        if random.random() > self.nsp_ratio:
            while 1:
                senB_idx = random.randint(1, self.num_lines)
                senB = linecache.getline(self.corpus_path, senB_idx)
                if senB_idx != ids + 1:
                    break
            isNext = 0
        else:
            senB = linecache.getline(self.corpus_path, ids + 1)
            isNext = 1
        seqA, seqB = tokenizer.encode_batch([senA, senB])
        seqA, seqB = seqA.ids, seqB.ids
        len_senA = len(seqA)
        len_senB = len(seqB)
        # sentence A가 max_len 보다 길 경우
        if len_senA > self.max_len - 3: 
            len_senA = self.max_len - 3
            seqA = seqA[:self.max_len - 3]
        # sentence A + sentence B가 max_len보다 긴 경우
        token_much = (len_senA + len_senB + 3) - self.max_len 
        if token_much > 0:
            len_senB = len_senB - token_much
            seqB = seqB[:-token_much] 
        len_seq = len_senA + len_senB
        seq_ids = [_ for _ in range(len_seq)]
        
        def random_choice_with_prob(lst, p):
            k = round(len(lst) * p)
            return random.choices(lst, k=k)
        
        mask = random_choice_with_prob(seq_ids, self.mask_ratio)
        mask_mask = random_choice_with_prob(mask, 0.8)
        mask_random = random_choice_with_prob(list(set(mask).difference(mask_mask)), 0.5)
        mask_itself = set(mask).difference(mask_mask).difference(mask_random)
        
        def get_random_token():
            # random token을 추출하는 코드
            _special_token_id = max(_cls, _sep, _mask)
            random_token = random.randint(_special_token_id + 1 , self.vocab_size) 
            return random_token
        
        def replace_token(ids, token):
        # mask로 선택된 token 하나를 ids를 참고해서 바꾸는 코드 
            if ids in mask_mask:
                return _mask
            elif ids in mask_random:
                replaced = get_random_token()
                return replaced
            elif ids in mask_itself:
                return token
            return token
        
        tokens = seqA + seqB
        replaced_tokens = [replace_token(idx, token) for idx, token in enumerate(tokens)]
     
        ids = [_cls] + tokens[:len_senA] + [_sep] + tokens[len_senA:] + [_sep]
        replaced_tokens = [_cls] + replaced_tokens[:len_senA] + [_sep] + replaced_tokens[len_senA:] + [_sep]
        len_ids = len(ids)
        mask = [m + 1 if m < len_senA else m + 2 for m in mask] # cls, sep 토큰 추가 돼서
        mask_ids = [1 if ids in mask else 0 for ids in range(len_ids)] 
        segment_ids = [0 for _ in range(len_senA + 2)] + [1 for _ in range(len_senB + 1)] 
        return torch.Tensor(ids).long(), torch.Tensor(mask_ids).long(), torch.Tensor(replaced_tokens).long(), torch.Tensor(segment_ids).long(), torch.Tensor([isNext]).long()
    
def pad_collate(batch):
    ids, mask_ids, replaced_ids, segment_ids, nsp = zip(*batch)
    named_tuple = namedtuple('data', ['ids', 'mask_ids', 'replaced_ids', 'segment_ids', 'nsp'])
    ids_pad = pad_sequence(ids, batch_first=True, padding_value=0) # 하드코딩 고쳐야함
    mask_ids_pad = pad_sequence(mask_ids, batch_first=True, padding_value=0)
    replaced_ids_pad = pad_sequence(replaced_ids, batch_first=True, padding_value=0)
    segment_ids_pad = pad_sequence(segment_ids, batch_first=True, padding_value=0)
    nsp_pad = pad_sequence(nsp, batch_first=True, padding_value=0)
    return named_tuple(ids_pad, mask_ids_pad, replaced_ids_pad, segment_ids_pad, nsp_pad)   
        

if __name__ == '__main__':
    bd = BERT_Dataset('/home/long8v/torch_study/paper/file/bert/bert.txt',
                      '/home/long8v/torch_study/paper/file/bert/vocab.json',
                     max_len=128,
                     nsp_ratio=0.5)
    
    def decode_from_tensor(ids):
        print(bd.tokenizer.decode(ids.tolist(), skip_special_tokens=False))
        
    print(bd.tokenizer.token_to_id('[PAD]'),bd.tokenizer.token_to_id('[UNK]'),bd.tokenizer.token_to_id('[CLS]'),
         bd.tokenizer.token_to_id('[SEP]'),bd.tokenizer.token_to_id('[EOD]'))
    for ids, mask_ids, replaced_tokens, segment_ids, isnext in bd:
        print(ids)
        decode_from_tensor(ids)
        print(mask_ids)
        decode_from_tensor(replaced_tokens)
        print(segment_ids)
        print(isnext)
        for i, mi, rep in zip(ids, mask_ids, replaced_tokens):
            if mi:
                print(f'''{bd.tokenizer.decode([i], skip_special_tokens=False)}-> {bd.tokenizer.decode([rep], skip_special_tokens=False)}''')
        break
        
    print('data loader..')
    for batch in DataLoader(bd, batch_size=16, collate_fn=pad_collate):
        print(batch.ids.shape)
        print(batch.ids)
        print(torch.sum(batch.mask_ids == 1))
        print(batch.replaced_ids.shape)
        print(batch.segment_ids.shape)
        print(batch.nsp.shape)
        break