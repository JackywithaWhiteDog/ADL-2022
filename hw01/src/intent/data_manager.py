from typing import List, Union, Tuple
import json

import torch
from torch import Tensor
from torch.utils import data

from src.utils.data_manager import DataManager
from src.intent.preprocess import filter_marks
from src.intent.dataset import IntentDataset

class IntentDataManager(DataManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_data(
        self,
        raw_data: List[dict],
        with_intents: bool=False
    ) -> Union[Tuple[Tensor, List[str], Tensor, Tensor], Tuple[Tensor, List[str], Tensor]]:
        x = [filter_marks(d["text"]).split() for d in raw_data]
        ids = [d["id"] for d in raw_data]
        length = torch.tensor([len(text) for text in x])
        x = torch.tensor(self.vocab.encode_batch(x, max_len=self.max_len))
        if not with_intents:
            return x, ids, length
        y = torch.tensor([self.intent2idx[d["intent"]] for d in raw_data])
        return x, ids, length, y

    def get_train_dataloader(self, with_eval=False, shuffle=True) -> data.DataLoader:
        assert self.data_dir is not None
        data_path = self.data_dir / "train.json"
        raw_data = json.loads(data_path.read_text())
        if with_eval:
            eval_data_path = self.data_dir / "eval.json"
            raw_data.extend(json.loads(eval_data_path.read_text()))
        x, ids, length, y = self.parse_data(raw_data, with_intents=True)
        dataset = IntentDataset(x, ids, length, y)
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )
        return dataloader

    def get_valid_dataloader(self) -> data.DataLoader:
        assert self.data_dir is not None
        data_path = self.data_dir / "eval.json"
        raw_data = json.loads(data_path.read_text())
        x, ids, length, y = self.parse_data(raw_data, with_intents=True)
        dataset = IntentDataset(x, ids, length, y)
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return dataloader

    def get_test_dataloader(self) -> data.DataLoader:
        assert self.test_file is not None
        raw_data = json.loads(self.test_file.read_text())
        x, ids, length = self.parse_data(raw_data, with_intents=False)
        dataset = IntentDataset(x, ids, length)
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return dataloader
