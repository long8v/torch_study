import mecab
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from utils import *
import sys

sys.path.append("../source")
from txt_cleaner.clean.master import MasterCleaner
from txt_cleaner.utils import *
from torch8text.data import Vocab, Field, LabelField

import pickle


class ELMoDataset(Dataset):
    def __init__(self, src, token_field, chr_field, token_max_len, chr_max_len):
        self.src = src
        self.token_field = token_field
        self.chr_field = chr_field
        self.named_tuple = namedtuple("data", ["src_chr", "trg"])
        self.token_max_len = token_max_len
        self.chr_max_len = chr_max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.named_tuple(
            self.idx_process(idx, is_char=True), self.idx_process(idx)[1:]
        )

    def idx_process(self, idx, is_char=False):
        data = self.src[idx]
        tokenize_data = self.token_field.preprocess(data)[: self.token_max_len]
        if is_char:
            chrs = self.chr_field.preprocess(tokenize_data)
            pad_chrs = self.chr_field.pad_process(chrs, max_len=self.chr_max_len)
            return pad_chrs
        return torch.Tensor(self.token_field.vocab.stoi(tokenize_data)).long()


class PetitionDataset:
    def __init__(self, config):
        self.config = config["DATA"]
        print(self.config)
        self.mecab_tokenizer = mecab.MeCab()
        self.cleaner = MasterCleaner(
            {"minimum_space_count": self.config["MINIMUM_SPACE_COUNT"]}
        )
        self.token_field = Field(
            tokenize=self.tokenize_pos,
            preprocessing=lambda e: self.cleaner.cleaning(e),
            init_token=False,
            eos_token=False,
            max_len=self.config["TOKEN_MAX_LEN"],
            min_freq=self.config["TOKEN_MIN_FREQ"],
        )
        self.chr_field = Field(
            tokenize=list,
            init_token=False,
            eos_token=False,
            max_len=self.config["CHR_MAX_LEN"],
            min_freq=self.config["CHR_MIN_FREQ"],
        )

    def __call__(self, corpus):
        sent_corpus = [
            sent
            for text in corpus
            for sent in sent_tokenize(text)
            if self.cleaner.cleaning(sent)
        ]
        self.sent_corpus = sent_corpus
        self.token_field.build_vocab(sent_corpus)
        self.chr_field.build_vocab(sent_corpus)
        return ELMoDataset(
            sent_corpus,
            self.token_field,
            self.chr_field,
            token_max_len=self.token_field.max_len,
            chr_max_len=self.chr_field.max_len,
        )

    def tokenize_pos(self, inp):
        if type(inp) == str:
            return self.mecab_tokenizer.morphs(inp)
        if type(inp) == list:
            return [self.tokenize_pos(i) for i in inp]

    def see_process(self, data):
        token_data = self.token_field.preprocess(data)
        print(token_data)
        token_chr_data = self.chr_field.preprocess(token_data)
        print(token_chr_data)
        process_chr = self.chr_field.pad_process(
            token_chr_data, max_len=self.chr_field.max_len
        )
        print(process_chr)


def pad_collate(batch):
    (src_chr, trg) = zip(*batch)
    named_tuple = namedtuple("data", ["src_chr", "trg"])
    src_chr_pad = pad_sequence(src_chr, batch_first=True, padding_value=0)
    trg_pad = pad_sequence(trg, batch_first=True, padding_value=0)
    return named_tuple(src_chr_pad, trg_pad)


if __name__ == "__main__":
    with open("../data/petitions.p", "rb") as f:
        corpus = pickle.load(f)
    config = read_yaml("../config.yaml")
    pet_ds = PetitionDataset(config)
    dl = pet_ds(corpus)
    for original, _ in zip(pet_ds.sent_corpus, dl):
        print(original)
        #         pet_ds.see_process(original)
        #         print(_.trg)
        print(pet_ds.token_field.vocab.itos(_.trg.tolist()))
        break
