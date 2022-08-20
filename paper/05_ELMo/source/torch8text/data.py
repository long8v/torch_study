from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
from .utils import *


class Vocab:
    def __init__(self, min_freq=0):
        self.min_freq = min_freq

    def __call__(
        self, sentence_list, init_token="<SOS>", eos_token="<EOS>", is_target=False
    ):
        self.stoi_dict = defaultdict(lambda: 1)
        self.stoi_dict["<UNK>"] = 1
        self.stoi_dict["<PAD>"] = 0
        if init_token:
            self.stoi_dict[init_token] = len(self.stoi_dict)  # 2
        if eos_token:
            self.stoi_dict[eos_token] = len(self.stoi_dict)  # 3 if we have init token
        self.special_tokens = list(self.stoi_dict)[:]
        self.special_tokens_idx = list(self.stoi_dict.values())[:]

        if type(sentence_list[0]) is list:
            all_tokens = [token for sentence in sentence_list for token in sentence]
        else:
            all_tokens = sentence_list
        all_tokens = [token for token in all_tokens if token not in self.stoi_dict]  #
        self.token_counter = Counter(all_tokens).most_common()
        token_counter = [
            word for word, count in self.token_counter if count > self.min_freq
        ]

        _index = len(self.stoi_dict)  # get number of special dict

        for num, word in enumerate(token_counter):
            self.stoi_dict[word] = num + _index  # start with _index

        self.itos_dict = {v: k for k, v in self.stoi_dict.items()}

    def build_from_dict(self, dict_obj):
        self.stoi_dict = defaultdict(lambda: 1)
        self.stoi_dict.update(dict_obj)
        self.itos_dict = {v: k for k, v in self.stoi_dict.items()}

    def stoi(self, tokens):
        if type(tokens) == str:
            tokens = [tokens]
        return [self.stoi_dict[token] for token in tokens]

    def itos(self, indices):
        if type(indices) != list:
            indices = [indices]
        return " ".join(
            [
                self.itos_dict[int(index)]
                for index in indices
                if self.itos_dict[index] != "<PAD>"
            ]
        )

    def __len__(self):
        return len(self.stoi_dict)


class Field:
    def __init__(
        self,
        tokenize=lambda e: e.split(),
        init_token="<SOS>",
        eos_token="<EOS>",
        preprocessing=None,
        lower=False,
        reverse=False,
        max_len=999,
        min_freq=0,
    ):
        self.tokenize = tokenize
        self.init_token = init_token
        self.eos_token = eos_token
        self.lower = lower
        self.reverse = reverse
        self.preprocessing = preprocessing
        self.vocab = None
        self.pad = lambda data, pad_num: nn.ConstantPad2d((0, pad_num), 0)(data)
        self.max_len = max_len
        self.min_freq = min_freq

    def build_vocab(self, data):
        self.vocab = Vocab(self.min_freq)
        self.vocab(self.preprocess(data))

    def build_vocab_from_dict(self, dict_obj):
        self.vocab = Vocab()
        self.vocab.build_from_dict(dict_obj)

    def preprocess(self, data):
        if type(data) == str:
            pass
        else:
            return [self.preprocess(d) for d in data]

        if self.lower:
            data = data.lower()

        if self.preprocessing:
            try:
                data = self.preprocessing(data)
            except:
                print(data)
        tokenized_data = self.tokenize(data)
        if self.reverse:
            tokenized_data = tokenized_data[::-1]
        if self.init_token:
            tokenized_data = [self.init_token] + tokenized_data
        if self.eos_token:
            tokenized_data = tokenized_data + [self.eos_token]
        return tokenized_data[: self.max_len]

    def process(self, data):
        return self.vocab.stoi(data)

    def pad_process(self, data, max_len):
        d_list = []
        for d in data:
            process_d = torch.tensor(self.process(d))
            pad_d = self.pad(process_d, max_len - len(process_d)).unsqueeze(0)
            d_list.append(pad_d)
        return torch.cat((d_list), 0)


class LabelField:
    def __init__(self, pad_token=None):
        self.pad_token = pad_token

    def build_vocab(self, data):
        self.vocab = Vocab()
        category = set(data)
        idx = 0
        category_dict = {}
        if self.pad_token:
            category_dict[idx] = self.pad_token
            idx += 1
        for cat in category:
            category_dict[idx] = cat
            idx += 1
        self.vocab.itos_dict = category_dict
        self.vocab.stoi_dict = {word: idx for idx, word in self.vocab.itos_dict.items()}

    def process(self, data):
        return self.vocab.stoi(data)
