from typing import List, Union, Tuple, Dict
import json

import torch
from torch import Tensor
from torch.utils import data

from src.utils.data_manager import DataManager
from src.utils.dataset import TokenDataset
from src.utils.vocab import pad_to_len

class SlotDataManager(DataManager):
    def __init__(self, *args, **kwargs):
        super(SlotDataManager, self).__init__(target="tag", *args, **kwargs)
        self.empty_tag_idx = -1

    @property
    def tag2idx(self) -> Dict[str, int]:
        return self.target2idx

    @property
    def idx2tag(self) -> Dict[int, str]:
        return self.idx2target

    def parse_data(
        self,
        raw_data: List[dict],
        with_tags: bool=False
    ) -> Union[Tuple[Tensor, List[str], Tensor, Tensor], Tuple[Tensor, List[str], Tensor]]:
        x = [
            [token.lower() for token in d["tokens"]]
            for d in raw_data
        ]
        ids = [d["id"] for d in raw_data]
        length = torch.tensor([len(tokens) for tokens in x])
        x = torch.tensor(self.vocab.encode_batch(x, max_len=self.max_len))
        if not with_tags:
            return x, ids, length
        y = torch.tensor(pad_to_len(
            seqs=[
                [self.tag2idx[tag] for tag in d["tags"]]
                for d in raw_data
            ],
            to_len=self.max_len,
            padding=self.empty_tag_idx
        ))
        return x, ids, length, y

    def get_train_dataloader(self, with_eval=False, shuffle=True) -> data.DataLoader:
        assert self.data_dir is not None
        data_path = self.data_dir / "train.json"
        raw_data = json.loads(data_path.read_text())
        if with_eval:
            eval_data_path = self.data_dir / "eval.json"
            raw_data.extend(json.loads(eval_data_path.read_text()))
        x, ids, length, y = self.parse_data(raw_data, with_tags=True)
        dataset = TokenDataset(x, ids, length, y)
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
        x, ids, length, y = self.parse_data(raw_data, with_tags=True)
        dataset = TokenDataset(x, ids, length, y)
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
        x, ids, length = self.parse_data(raw_data, with_tags=False)
        dataset = TokenDataset(x, ids, length)
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return dataloader
