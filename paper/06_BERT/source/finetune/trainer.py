import mlflow
import sys

sys.path.append("~/torch_study/paper/06_BERT/source")
from utils import *
from finetune.ner_bert import *
from finetune.dataset import *
import torch
import dill
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader


class NER_BERT_trainer(pl.LightningModule):
    def __init__(self, config):
        super(NER_BERT_trainer, self).__init__()
        self.config = config
        data_config = config["data"]

        dataset = NER_Dataset(data_config["src"], data_config["vocab"])
        valid_dataset = NER_Dataset(data_config["src_valid"], data_config["vocab"])

        self.pad_idx = dataset.tokenizer.token_to_id("[PAD]")
        dataloader = DataLoader(
            dataset,
            data_config["batch_size"],
            collate_fn=lambda batch: pad_collate(batch, self.pad_idx),
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            data_config["batch_size"],
            collate_fn=lambda batch: pad_collate(batch, self.pad_idx),
        )

        print("before special token", dataset.tokenizer.get_vocab_size())
        dataset.tokenizer.add_special_tokens(["[SEP]", "[CLS]", "[MASK]", "[EOD]"])
        self.vocab_size = dataset.tokenizer.get_vocab_size()
        print("after special token", self.vocab_size)

        self.output_dim = dataset.output_dim
        self.ner_bert = NER_BERT(
            self.config, self.vocab_size, self.output_dim, self.pad_idx
        )
        device = config["train"]["device"]
        self.ner_bert.to(device)
        self.ner_bert.train()
        self.ner_bert.zero_grad()
        self.ner_bert.fcn.apply(self.initialize_weights)

        if device == "cuda":
            self.gpus = 1
        else:
            self.gpus = 0

        artifact_path = mlflow.mlflow.get_artifact_uri()
        artifact_path = artifact_path.replace("file://", "")
        print("////////////////arftifact_path", artifact_path)
        checkpoint_callback = ModelCheckpoint(
            dirpath=artifact_path, monitor="valid_macro_f1"
        )

        # Add your callback to the callbacks list
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=self.config["train"]["n_epochs"],
            progress_bar_refresh_rate=10,
            gpus=self.gpus,
        )

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()

        # Train the model
        mlflow.end_run()  # 이전에 돌아가고 있던거 끄기
        with mlflow.start_run() as run:
            for key, value in config.items():
                mlflow.log_param(key, value)
            trainer.fit(self.ner_bert, dataloader, valid_dataloader)

    # https://github.com/GyuminJack/torchstudy/blob/main/06Jun/BERT/src/trainer.py
    def initialize_weights(self, m):
        for name, param in m.named_parameters():
            if ("fc" in name) or ("embedding" in name):
                if "bias" in name:
                    torch.nn.init.zeros_(param.data)
                else:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
            elif "layer_norm" in name:
                if "bias" in name:
                    torch.nn.init.zeros_(param.data)
                else:
                    torch.nn.init.constant_(param.data, 1.0)


if __name__ == "__main__":
    config_file = "~/torch_study/paper/06_BERT/config_finetune.yaml"
    config = read_yaml(config_file)
    print("train started..")
    trainer = NER_BERT_trainer(config)
