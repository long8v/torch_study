import os
import argparse
import torch
from sklearn.model_selection import KFold
import torch.optim as optim
from source.dataloader import *
from source.trainer import *
from source.model import *
from tqdm import tqdm
from torch.utils.data.dataset import Subset


def main():
    dataset = CNNDataset(args.path, args.w2v_path)
    kf = KFold(n_splits=args.kfold, shuffle=True)
    kf_splitted = kf.split(dataset)
    kf_list_index = list(kf_splitted)
    kf_folded = [
        (Subset(dataset, train_idx), Subset(dataset, test_idx))
        for train_idx, test_idx in kf_list_index
    ]
    kth_fold = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"your  deivce is {device}")

    kfold_accuracy_list = []
    for train_ds, valid_ds in kf_folded:
        print(f"{kth_fold}th fold..")
        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=args.batch_size,
            collate_fn=dataset.pad_collate,
            shuffle=True,
        )

        valid_dl = DataLoader(
            dataset=valid_ds,
            batch_size=args.batch_size,
            collate_fn=dataset.pad_collate,
            shuffle=True,
        )
        model = CNN(
            dataset.pretrained_embedding,
            len(dataset.vocab.stoi_dict),
            dataset.embedding_dim,
            args.n_filters,
            args.filters_sizes,
            args.output_dim,
            args.dropout,
            dataset.vocab.stoi_dict["<PAD>"],
        )
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        kfold_accuracy_list.append(
            logging_train(
                model,
                train_dl,
                valid_dl,
                optimizer,
                criterion,
                args.path,
                args.n_epochs,
                args.model_name,
            )
        )
        kth_fold += 1
    print("-------- Done -------")
    print(f"{args.model_name} K-fold accuracy mean : {np.mean(kfold_accuracy_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Builder")
    parser.add_argument("-k", "--kfold", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=50)
    parser.add_argument("-p", "--path", type=str, default="/home/long8v")
    parser.add_argument(
        "-w",
        "--w2v_path",
        type=str,
        default="~/torch_study/data/GoogleNews-vectors-negative300.bin.gz",
    )
    parser.add_argument("-n", "--n_filters", type=int, default=100)
    parser.add_argument("-f", "--filters_sizes", type=list, default=[3, 4, 5])
    parser.add_argument("-o", "--output_dim", type=int, default=2)
    parser.add_argument("-d", "--dropout", type=float, default=0.5)
    parser.add_argument("-epochs", "--n_epochs", type=int, default=10)
    parser.add_argument("-m", "--model_name", type=str, default="multi-channel")
    args = parser.parse_args()

    main()
