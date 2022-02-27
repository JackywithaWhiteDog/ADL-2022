import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_lightning import LightningModule
from collections import Sequence

from src import logger

class IntentClassifier(LightningModule):
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
        super(IntentClassifier, self).__init__()
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
        fc_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, fc_hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_class),
        )
        # Training
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()
        # logger.info(f"Bidirectional: {self.bidirectional}")
        self.save_hyperparameters()

    def acc(self, pred, target):
        return (pred.argmax(dim=1) == target).float().mean()

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

        if self.bidirectional:
            forward_features, backward_features = torch.chunk(output_features, 2, dim=2)
            last_features = torch.cat((forward_features[:, -1, :], backward_features[:, 0, :]), dim=1)
        else:
            last_features = output_features[:, -1, :]

        output = self.fc(last_features)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def training_step(self, batch, batch_idx):
        x, length, y = batch
        output = self(x, length)
        loss = self.loss(input=output, target=y)
        acc = self.acc(pred=output, target=y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # tensorboard_logs = {'acc': {'train': acc }, 'loss':{'train': loss.detach() } }
        return loss

    def validation_step(self, batch, batch_idx):
        x, length, y = batch
        output = self(x, length)
        loss = self.loss(input=output, target=y)
        acc = self.acc(pred=output, target=y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # tensorboard_logs = {'acc': {'val': acc }, 'loss':{'val': loss.detach() } }
        return loss

    def predict_step(self, batch, batch_idx):
        x, length = batch
        output = self(x, length)
        pred = output.argmax(dim=1)
        return pred