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
        print('senA', senA, 'senB', senB)
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
            random_token = random.randint(0, self.vocab_size) # speecial token 빼야험
            return random_token
        
        def replace_token(ids, token):
        # mask로 선택된 token 하나를 ids를 참고해서 바꾸는 코드 
            if ids in mask_mask:
                return tokenizer.token_to_id(self.mask_token)
            elif ids in mask_random:
                replaced = get_random_token()
                return replaced
            elif ids in mask_itself:
                return token
            return token
        
        tokens = seqA + seqB
        replaced_tokens = [replace_token(idx, token) for idx, token in enumerate(tokens)]
        sep, cls = tokenizer.token_to_id('[SEP]'), tokenizer.token_to_id('[CLS]')
        ids = [cls] + tokens[:len_senA] + [sep] + tokens[len_senA:] + [sep]
        replaced_tokens = [cls] + replaced_tokens[:len_senA] + [sep] + replaced_tokens[len_senA:] + [sep]
        len_ids = len(ids)
        mask_ids = [1 if ids in mask else 0 for ids in range(len_ids)] 
        mask_ids = [0] + mask_ids[:len_senA] + [0] + mask_ids[len_senA:] + [0]
        segment_ids = [0 for _ in range(len_senA + 2)] + [1 for _ in range(len_senB + 1)] 
        return torch.Tensor(ids).long(), torch.Tensor(mask_ids).long(), torch.Tensor(replaced_tokens).long(), torch.Tensor(segment_ids).long()
    
 
        

if __name__ == '__main__':
    bd = BERT_Dataset('/home/long8v/torch_study/paper/file/bert/bert.txt',
                      '/home/long8v/torch_study/paper/file/bert/vocab.json',
                     256,
                     0.5)
    def decode_from_tensor(ids):
        print(bd.tokenizer.decode(ids.tolist(), skip_special_tokens=False))
        
    for ids, mask_ids, replaced_tokens, segment_ids in bd:
        decode_from_tensor(ids)
        print(mask_ids)
        decode_from_tensor(replaced_tokens)
        print(segment_ids)
        break