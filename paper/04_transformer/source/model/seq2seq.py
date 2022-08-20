import mlflow
import pytorch_lightning as pl
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient
import torch.nn as nn
from .labelsmoothing import *
import torch.optim as optim
from sacrebleu import corpus_bleu, sentence_bleu


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        src_pad_idx,
        trg_pad_idx,
        device,
        lr,
        trg_field,
        output_dim,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.new_device = device
        self.lr = lr
        self.criterion = LabelSmoothingLoss_aftersoftmax(output_dim)
        self.trg_field = trg_field

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.new_device)
        ).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention

    def training_step(self, batch, batch_nb):
        src, trg = batch.src, batch.trg
        src, trg = src.to(self.new_device), trg.to(self.new_device)
        output, _ = self(src, trg[:, :-1])
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        bleu = self.get_bleu_score(output, trg[:, :-1], self.trg_field)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        loss = self.criterion(output, trg)

        self.log("train_loss", loss, on_step=True)
        self.log("train_bleu", bleu, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch.src, batch.trg
        src, trg = src.to(self.new_device), trg.to(self.new_device)
        output, _ = self(src, trg[:, :-1])
        bleu = self.get_bleu_score(output, trg[:, :-1], self.trg_field)

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        loss = self.criterion(output, trg)

        self.log("valid_loss", loss, on_step=True)
        self.log("valid_bleu", bleu, on_step=True)
        return loss

    def get_bleu_score(self, output, trg, trg_field):
        """
        bleu 코드 다시 짜야함!        
        """

        def get_speical_token(field):
            def get_stoi(idx):
                return field.vocab.stoi[idx]

            return [
                get_stoi(field.pad_token),
                get_stoi(field.unk_token),
                get_stoi(field.eos_token),
            ]

        def get_itos_str(tokens, field):
            ignore_idx = get_speical_token(field)
            return " ".join(
                [field.vocab.itos[token] for token in tokens if token not in ignore_idx]
            )

        def get_itos_batch(tokens_batch, field):
            return [get_itos_str(batch, field) for batch in tokens_batch]

        with torch.no_grad():
            # output shape :
            output_token = output.argmax(-1)
            print("output shape", output.shape)
        #         output_token = output_token.permute(1, 0)
        #         trg = trg.permute(1, 0)
        system = get_itos_batch(output_token, trg_field)
        refs = get_itos_batch(trg, trg_field)
        print("len system:", len(system))
        print("len refs", len(refs))
        print("system: ", system[1][:100])
        print("refs: ", refs[1][:100])
        bleu = corpus_bleu(system, [refs], force=True).score
        return bleu

    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [
            f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")
        ]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
