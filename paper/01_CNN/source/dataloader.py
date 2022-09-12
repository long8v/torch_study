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
        self.stoi_dict = defaultdict(lambda: 1)
        self.stoi_dict["<PAD>"] = 0
        self.stoi_dict["<UNK>"] = 1
        _index = 2
        for sentence in sentence_list:
            tokens_list = sentence
            for word in tokens_list:
                if word in self.stoi_dict:
                    pass
                else:
                    self.stoi_dict[word] = _index
                    _index += 1
        self.itos_dict = {v: k for k, v in self.stoi_dict.items()}

    def stoi(self, token_list):
        return [self.stoi_dict[word] for word in token_list]

    def itos(self, indices):
        return " ".join(
            [
                self.itos_dict[index]
                for index in indices
                if self.itos_dict[index] != "<PAD>"
            ]
        )


class CNNDataset:
    def __init__(self, path, w2v_path):
        data = self.load_data(path)
        zipped_data = list(zip(*data))

        self.text = zipped_data[0]
        self.text = [self.clean_str(sen) for sen in self.text]
        self.text = [[word for word in self.tokenizer(sen)] for sen in self.text]
        self.label = zipped_data[1]

        self.vocab = Vocab()
        self.vocab.build_vocabs(self.text)
        self.pad_index = self.vocab.stoi_dict["<PAD>"]
        self.w2v = self.load_word2vec(w2v_path)
        self.embedding_dim = self.w2v.vector_size
        self.pretrained_embedding = self.get_pretrained_embeddings()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample_label = self.label[idx]
        sample_text = self.text[idx]
        sample_text = self.vocab.stoi(sample_text)
        return torch.Tensor(sample_text).long(), sample_label

    def load_data(self, path):
        with open(
            f"{path}/CNN_sentence/rt-polarity.pos", "r", encoding="ISO-8859-1"
        ) as f:
            pos = f.readlines()
        with open(
            f"{path}/CNN_sentence/rt-polarity.neg", "r", encoding="ISO-8859-1"
        ) as f:
            neg = f.readlines()
        pos = [(p, 1) for p in pos]
        neg = [(n, 0) for n in neg]
        return pos + neg

    def tokenizer(self, sentence):
        return sentence.split()

    def load_word2vec(self, w2v_path):
        return KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    def get_pretrained_embeddings(self):
        pretrained_embedding = []
        for word in self.vocab.stoi_dict:
            if word in self.w2v:
                pretrained_embedding.append(self.w2v[word])
            else:
                pretrained_embedding.append(
                    np.random.uniform(-0.25, 0.25, self.embedding_dim)
                )
        return torch.from_numpy(np.array(pretrained_embedding))

    def pad_collate(self, batch):
        (xx, yy) = zip(*batch)
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        return xx_pad, yy

    def clean_str(self, string, TREC=False):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " 's", string)
        string = re.sub(r"\'ve", " 've", string)
        string = re.sub(r"n\'t", " n't", string)
        string = re.sub(r"\'re", " 're", string)
        string = re.sub(r"\'d", " 'd", string)
        string = re.sub(r"\'ll", " 'll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip() if TREC else string.strip().lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Builder")
    parser.add_argument("-b", "--batch_size", type=int, default=10)
    parser.add_argument("-p", "--path", type=str, default="/home/long8v")
    parser.add_argument(
        "-w",
        "--w2v_path",
        type=str,
        default="/home/long8v/Downloads/GoogleNews-vectors-negative300.bin.gz",
    )
    args = parser.parse_args()

    dataset = CNNDataset(args.path, args.w2v_path)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.pad_collate
    )

    for i in data_loader:
        print(i)
        break
