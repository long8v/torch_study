import mecab
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import sys

sys.path.append("../")

from txt_cleaner.clean.master import MasterCleaner
from utils import *
from txt_cleaner.utils import *
from torch8text.data import Vocab, Field, LabelField
import pickle


class ELMoDataset_finetune(Dataset):
    def __init__(
        self, src, trg, token_field, chr_field, label_field, token_max_len, chr_max_len
    ):
        self.src = src
        self.trg = list(trg)
        self.token_field = token_field
        self.chr_field = chr_field
        self.label_field = label_field
        self.named_tuple = namedtuple("data", ["src_token", "src_chr", "trg"])
        self.token_max_len = token_max_len
        self.chr_max_len = chr_max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.named_tuple(
            self.idx_process(idx),
            self.idx_process(idx, is_char=True),
            self.label_idx_process(idx),
        )

    def label_idx_process(self, idx):
        data = self.trg[idx]
        return torch.Tensor(self.label_field.process(data)).long()

    def idx_process(self, idx, is_char=False):
        data = self.src[idx]
        tokenize_data = self.token_field.preprocess(data)[: self.token_max_len]
        if is_char:
            chrs = self.chr_field.preprocess(tokenize_data)
            pad_chrs = self.chr_field.pad_process(chrs, max_len=self.chr_max_len)
            return pad_chrs
        return torch.Tensor(self.token_field.vocab.stoi(tokenize_data)).long()


class PetitionDataset_finetune:
    def __init__(self, config):
        self.config = config["DATA"]
        print(self.config)
        self.mecab_tokenizer = mecab.MeCab()
        self.cleaner = MasterCleaner(
            {"minimum_space_count": self.config["MINIMUM_SPACE_COUNT"]}
        )
        self.token_field = Field(
            tokenize=lambda e: e.split(),  # self.tokenize_pos,
            preprocessing=None,
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
        self.label_field = LabelField(dtype=torch.float)

    def __call__(self, corpus_category, token_stoi_dict, chr_stoi_dict):
        # corpus : [(corpus, category), ]
        clean_data = [
            (self.cleaner.cleaning(corpus), category)
            for corpus, category in corpus_category
        ]
        corpus = [c for c, _ in clean_data if c]
        category = [category for _, category in clean_data if _]

        self.corpus = corpus
        self.token_field.build_vocab_from_dict(token_stoi_dict)
        self.chr_field.build_vocab_from_dict(chr_stoi_dict)
        self.label_field.build_vocab(category)
        return ELMoDataset_finetune(
            corpus,
            category,
            self.token_field,
            self.chr_field,
            self.label_field,
            token_max_len=self.token_field.max_len,
            chr_max_len=self.chr_field.max_len,
        )

    def tokenize_pos(self, inp):
        if type(inp) == str:
            return self.mecab_tokenizer.morphs(inp)
        if type(inp) == list:
            return [self.tokenize_pos(i) for i in inp]


def pad_collate_finetune(batch):
    (src_token, src_chr, trg) = zip(*batch)
    named_tuple = namedtuple("data", ["src_token", "src_chr", "trg"])
    src_token_pad = pad_sequence(src_token, batch_first=True, padding_value=0)
    src_chr_pad = pad_sequence(src_chr, batch_first=True, padding_value=0)
    trg = torch.tensor(trg)
    return named_tuple(src_token_pad, src_chr_pad, trg)


if __name__ == "__main__":
    with open("~/torch_study/data/ynat/train_tokenized.ynat", "r") as f:
        corpus = f.readlines()
    config = read_yaml("~/torch_study/paper/05_ELMo/config_finetune.yaml")
    corpus = [(txt.split("\t")[1], txt.split("\t")[0]) for txt in corpus]
    pet_ds = PetitionDataset_finetune(config)
    chr_dict = read_yaml(
        "~/torch_study/paper/05_ELMo/source/model/elmo_30051604/chr_dict.yaml"
    )
    token_dict = read_yaml(
        "~/torch_study/paper/05_ELMo/source/model/elmo_30051604/token_dict.yaml"
    )
    ds = pet_ds(corpus, chr_dict, token_dict)
    dl = DataLoader(ds, batch_size=1, collate_fn=pad_collate_finetune)
    print([c for c in pet_ds.corpus if "복사꽃" in c])
    for original, _ in zip(pet_ds.corpus, dl):
        print(original)
        print(pet_ds.label_field.vocab.itos(_.trg.tolist()))
        pass
