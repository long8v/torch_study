# https://keep-steady.tistory.com/37#recentEntries
# 
from torch.utils.data import Dataset, DataLoader
import random
import numpy.random
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
        return self.num_lines
    
    def __getitem__(self, ids):
        tokenizer = self.tokenizer
        _sep, _cls, _mask = tokenizer.token_to_id('[SEP]'), tokenizer.token_to_id('[CLS]'), tokenizer.token_to_id(self.mask_token)
        
        
        ids = ids + 1
        senA = linecache.getline(self.corpus_path, ids)
        if random.random() > self.nsp_ratio:
            while 1:
                senB_idx = random.randint(1, self.num_lines)
                senB = linecache.getline(self.corpus_path, senB_idx)
                if senB_idx == ids + 1 or senB.strip() == '<EOD>':
                    pass
                else:
                    break
            isNext = 0
        else:
            senB = linecache.getline(self.corpus_path, ids + 1)
            isNext = 1
        seqA, seqB = tokenizer.encode_batch([senA, senB])
        seqA, seqB = seqA.ids, seqB.ids
        len_senA = len(seqA)
        len_senB = len(seqB)
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
        mask_remain = set(mask).difference(mask_mask)
        mask_itself = random_choice_with_prob(list(mask_remain), 0.5)
        mask_remain = mask_remain.difference(mask_itself)
        mask_random = random_choice_with_prob(list(mask_remain), 0.5)
        
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
        mask_ids = [1 if ids in mask else 0 for ids in range(len_ids)] 
        mask_ids = [0] + mask_ids[:len_senA] + [0] + mask_ids[len_senA:] + [0]
        segment_ids = [0 for _ in range(len_senA + 2)] + [1 for _ in range(len_senB + 1)] 
        return torch.Tensor(ids).long(), torch.Tensor(mask_ids).long(), torch.Tensor(replaced_tokens).long(), torch.Tensor(segment_ids).long(), torch.Tensor([isNext]).long()
    
 
        

if __name__ == '__main__':
    bd = BERT_Dataset('/home/long8v/torch_study/paper/file/bert/bert.txt',
                      '/home/long8v/torch_study/paper/file/bert/vocab.json',
                     256,
                     0.5)
    def decode_from_tensor(ids):
        print(bd.tokenizer.decode(ids.tolist(), skip_special_tokens=False))
        
    for ids, mask_ids, replaced_tokens, segment_ids, isnext in bd:
        decode_from_tensor(ids)
        print(mask_ids)
        decode_from_tensor(replaced_tokens)
        print(segment_ids)
        print(isnext)
        break
        '''
[CLS] 현재 사대, 교대 등 교원양성학교들의 예비교사들이 임용절벽에 매우 힘들어 하고 있는 줄로 압니다. [SEP] 정부 부처에서는 영양사의 영양 교사 화, 폭발적인 영양 교사 채용, 기간제 교사, 영전강, 스강의 무기계약직화가 그들의 임용 절벽과는 전혀 무관한 일이라고 주장하고 있지만 조금만 생각해보면 전혀 설득력 없는 말이라고 생각합니다. [SEP]
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
        0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0])
[CLS] 현재 사대, 교대 등 교원양성학교들의 예비교사들이 임용절벽에 매우 힘들어 하고 있는 줄로 압니다. [SEP] 정부 부처에서는 영양사의 영양 교사 화, 폭발적인 영양 교사 채용, 기간제 교사, 영전강, 스강 [MASK] 무기계약직화가 그들의 [MASK] 절벽 [MASK]는 전혀 [MASK]한 [MASK] [MASK]라고 방글라데시하고 있지만 조금만 생각해보면 전혀 설득력 없는 말이라고 생각합니다 [MASK] [SEP]
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1])
tensor([1])
        '''