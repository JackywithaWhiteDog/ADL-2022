import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_lightning import LightningModule
from collections import Sequence

from src import logger

class SlotTagger(LightningModule):
    def __init__(
        self,
        embeddings: Tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        lr: float,
        weight_decay: float
    ):
        super(SlotTagger, self).__init__()
        # Architecture
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)
        embedding_dim = embeddings.shape[1]
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.fc_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fc_hidden_size, self.fc_hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fc_hidden_size, num_class),
        )
        # Training
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()
        # logger.info(f"Bidirectional: {self.bidirectional}")
        self.save_hyperparameters()

    def token_acc(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred.argmax(dim=1) == target).float().mean()

    def join_acc(self, pred: Tensor, target: Tensor, length: Tensor) -> Tensor:
        return torch.tensor([
            torch.all(sen_val[:sen_len].argmax(dim=1) == sen_tags[:sen_len])
            for sen_val, sen_tags, sen_len in zip(pred, target, length)
        ]).float().mean()

    def forward(self, x: Tensor, length: Tensor) -> Tensor:
        embeddings = self.embedding(x)
        packed_features = pack_padded_sequence(
            input=embeddings,
            lengths=length.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output_features, _ = self.rnn(packed_features)
        output_features, _ = pad_packed_sequence(
            sequence=packed_output_features,
            batch_first=True
        )
        flatten_features = output_features.view(-1, self.fc_hidden_size)
        flatten_output = self.fc(flatten_features)
        output = flatten_output.view(output_features.shape[0], output_features.shape[1], -1)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=False)

    def training_step(self, batch, batch_idx):
        x, length, y = batch
        output = self(x, length)
        flatten_output = torch.cat([
            sen_output[:sen_len, :]
            for sen_output, sen_len, in zip(output, length)
        ])
        flatten_y = torch.cat([
            sen_tags[:sen_len]
            for sen_tags, sen_len, in zip(y, length)
        ])
        loss = self.loss(input=flatten_output, target=flatten_y)
        token_acc = self.token_acc(pred=flatten_output, target=flatten_y)
        join_acc = self.join_acc(pred=output, target=y, length=length)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_token_acc", token_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_join_acc", join_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # tensorboard_logs = {'acc': {'train': acc }, 'loss':{'train': loss.detach() } }
        return loss

    def validation_step(self, batch, batch_idx):
        x, length, y = batch
        output = self(x, length)
        flatten_output = torch.cat([
            sen_output[:sen_len, :]
            for sen_output, sen_len, in zip(output, length)
        ])
        flatten_y = torch.cat([
            sen_tags[:sen_len]
            for sen_tags, sen_len, in zip(y, length)
        ])
        loss = self.loss(input=flatten_output, target=flatten_y)
        token_acc = self.token_acc(pred=flatten_output, target=flatten_y)
        join_acc = self.join_acc(pred=output, target=y, length=length)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_token_acc", token_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_join_acc", join_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # tensorboard_logs = {'acc': {'val': acc }, 'loss':{'val': loss.detach() } }
        return loss

    def predict_step(self, batch, batch_idx):
        x, length = batch
        output = self(x, length)
        pred = [
            sen_val[:sen_len].argmax(dim=1)
            for sen_val, sen_len in zip(output, length)
        ]
        return pred