from pathlib import Path
import pickle
import json
from typing import Optional

import torch
from torch.utils import data

from src import logger

class DataManager:
    def __init__(
        self,
        cache_dir: Path,
        max_len: int,
        batch_size: int,
        num_workers: int,
        data_dir: Optional[Path]=None,
        test_file: Optional[Path]=None,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_len = max_len if max_len > 0 else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.test_file = test_file

        self.load_vocab()
        self.load_intent2idx()
        self.load_embeddings()

    @property
    def num_class(self) -> int:
        return len(self.intent2idx) if hasattr(self, "intent2idx") else 0

    def load_vocab(self) -> None:
        vocab_path = self.cache_dir / "vocab.pkl"
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        logger.info(f"Vocab loaded from {str(vocab_path.resolve())}")

    def load_intent2idx(self) -> None:
        intent_idx_path = self.cache_dir / "intent2idx.json"
        self.intent2idx = json.loads(intent_idx_path.read_text())
        self.idx2intent = {
            idx: intent
            for intent, idx in self.intent2idx.items()
        }
        logger.info(f"Intent-2-Index loaded from {str(intent_idx_path.resolve())}")

    def load_embeddings(self) -> None:
        embeddings_path = self.cache_dir / "embeddings.pt"
        self.embeddings = torch.load(embeddings_path)
        logger.info(f"Embeddings loaded from {str(embeddings_path.resolve())}")

    def get_train_dataloader(self, with_eval=False, shuffle=True) -> data.DataLoader:
        return NotImplementedError

    def get_valid_dataloader(self) -> data.DataLoader:
        return NotImplementedError

    def get_test_dataloader(self) -> data.DataLoader:
        return NotImplementedError
