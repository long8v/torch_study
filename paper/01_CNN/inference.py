import os
import argparse
import torch
from sklearn.model_selection import KFold
import torch.optim as optim
from source.dataloader import *
from source.trainer import *
from source.model import *
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('-k', '--kfold', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-p', '--path', type=str, default='/home/long8v')
    parser.add_argument('-w', '--w2v_path', type=str, default='/home/long8v/Downloads/GoogleNews-vectors-negative300.bin.gz')
    parser.add_argument('-n', '--n_filters', type=int, defult=100)
    parser.add_argument('-f', '--filters_sizes', type=list, defult=[3, 4, 5])
    parser.add_argument('-o', '--output_dim', type=int, defult=2)
    parser.add_argument('-d', '--dropout', type=float, defult=0.5)   
    parser.add_argument('-n', '--n_epochs', type=int, defult=10)   
    args = parser.parse_args()

    dataset = CNNDataset(args.path, args.w2v_path)
    kf = KFold(n_splits=args.kfold)
    kf_splitted = kf.split(dataset)
    kf_list_index = list(kf_splitted)
    kf_folded = [(Subset(dataset, train_idx), Subset(dataset, test_idx)) 
                for train_idx, test_idx in kf_list_index]

    
    device = torch.device("cuda:0" if torch.cuda.is_available() 
    else "cpu")
    model.to_(device)

    for train_ds, valid_ds in tqdm(kf_folded):
        train_dl = DataLoader(dataset=train_ds, 
                            batch_size=args.batch_size,
                            collate_fn=dataset.pad_collate)

        valid_dl = DataLoader(dataset=valid_ds, 
                            batch_size=args.batch_size,
                            collate_fn=dataset.pad_collate)

        model = CNN(train_ds.pretrained_embedding, len(train_ds.vocab), train_ds.embedding_dim,
                    args.n_filters, args.output_dim, args.dropout, train_ds.vocab.stoi_dict['<PAD>'])

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        verbose_train(model, train_dl, valid_dl, optimizer, criterion, args.path, args.n_epoch)