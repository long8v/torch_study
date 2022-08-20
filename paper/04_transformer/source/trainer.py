import mlflow
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader
from .dataset import *
from .model.encoder import Encoder
from .model.decoder import Decoder
from .model.seq2seq import Seq2Seq
from .utils import *


class multi30k_trainer(pl.LightningModule):
    def __init__(self, config):
        super(multi30k_trainer, self).__init__()
        self.config = config
        bs = self.config["TRAIN"]["BATCH_SIZE"]
        device = self.config["TRAIN"]["DEVICE"]
        lr = self.config["TRAIN"]["LR"]
        multi30k = Multi30k_Dataset()
        self.train_iter, self.valid_iter, self.test_iter = multi30k(batch_size=bs)
        INPUT_DIM = len(multi30k.SRC.vocab)

        config_encoder = self.config["MODEL"]["ENCODER"]
        self.encoder = Encoder(
            INPUT_DIM,
            config_encoder["HID_DIM"],
            config_encoder["N_LAYERS"],
            config_encoder["N_HEADS"],
            config_encoder["PF_DIM"],
            config_encoder["DROPOUT"],
            device,
        )

        OUTPUT_DIM = len(multi30k.TRG.vocab)
        config_decoder = self.config["MODEL"]["DECODER"]
        self.decoder = Decoder(
            OUTPUT_DIM,
            config_decoder["HID_DIM"],
            config_decoder["N_LAYERS"],
            config_decoder["N_HEADS"],
            config_decoder["PF_DIM"],
            config_decoder["DROPOUT"],
            device,
        )

        SRC_PAD_IDX = multi30k.SRC.vocab.stoi[multi30k.SRC.pad_token]
        TRG_PAD_IDX = multi30k.TRG.vocab.stoi[multi30k.TRG.pad_token]
        self.seq2seq = Seq2Seq(
            self.encoder,
            self.decoder,
            SRC_PAD_IDX,
            TRG_PAD_IDX,
            device,
            lr,
            multi30k.TRG,
            OUTPUT_DIM,
        )

        self.seq2seq.to(device)
        self.seq2seq.train()
        self.seq2seq.zero_grad()
        self.seq2seq.apply(self.initialize_weights)

        trainer = pl.Trainer(
            max_epochs=config["TRAIN"]["N_EPOCHS"],
            progress_bar_refresh_rate=10,
            gpus=1,
            auto_lr_find=True,
        )

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()

        # Train the model
        mlflow.end_run()  # 이전에 돌아가고 있던거 끄기
        with mlflow.start_run() as run:
            mlflow.log_params(config)
            trainer.fit(self.seq2seq, self.train_iter, self.valid_iter)

    def initialize_weights(self, m):
        if hasattr(m, "weight"):
            if m.weight is None:
                print(m)  # weight가 None인 것들이 있음 -> crossentropy loss
            elif m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)


if __name__ == "__main__":
    config = read_yaml("~/torch_study/paper/04_transformer/config.yaml")
    multi30k_trainer(config)
