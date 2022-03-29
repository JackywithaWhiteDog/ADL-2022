from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Union, List, Optional
from itertools import chain

import numpy as np
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataManager:

    tokenizer: PreTrainedTokenizerBase
    context_path: Union[str, Path]
    train_path: Optional[Union[str, Path]] = None
    valid_path: Optional[Union[str, Path]] = None
    test_path: Optional[Union[str, Path]] = None
    max_seq_length: Optional[int] = None
    que_col: str = "question"
    para_col: str = "paragraphs"
    num_para: int = 4
    rel_col: str = "relevant"
    label_col: str = "labels"

    context: List[str] = field(init=False)
    train_dataset: Dataset = field(default=None, init=False)
    valid_dataset: Dataset = field(default=None, init=False)
    test_dataset: Dataset = field(default=None, init=False)

    def __post_init__(self):
        self.max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length) if self.max_seq_length is not None else self.tokenizer.model_max_length
        self.context = self.load_json(self.context_path)
        if self.train_path is not None:
            self.train_dataset = self.build_dataset(self.train_path)
        if self.valid_path is not None:
            self.valid_dataset = self.build_dataset(self.valid_path)
        if self.test_path is not None:
            self.test_dataset = self.build_dataset(self.test_path)

    def load_json(self, file_path: Union[str, Path]) -> List:
        result = []
        with open(file_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        return result

    def preprocess(self, instances):
        questions = [
            [question] * self.num_para
            for question in instances[self.que_col]
        ]
        paragraphs = [
            [
                self.context[idx] if idx < len(self.context) else ""
                for idx in paragraphs
            ]
            for paragraphs in instances[self.para_col]
        ]

        # Flatten
        questions = list(chain.from_iterable(questions))
        paragraphs = list(chain.from_iterable(paragraphs))

        # Tokenize
        tokenized_examples = self.tokenizer(
            questions,
            paragraphs,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False
        )

        # Un-flatten
        result = {
            k: [
                v[i: i+self.num_para]
                for i in range(0, len(v), self.num_para)
            ]
            for k, v in tokenized_examples.items()
        }

        if self.rel_col in instances:
            result[self.label_col] = [
                paragraphs.index(relevant)
                for paragraphs, relevant in zip(instances[self.para_col], instances[self.rel_col])
            ]

        return result

    def build_dataset(self, file_path: Union[str, Path]) -> Dataset:
        data = self.load_json(file_path)
        keys = data[0].keys()
        dataset = Dataset.from_dict({
            key: [item[key] for item in data]
            for key in keys
        })
        return dataset.map(
            self.preprocess,
            batched=True,
            num_proc=None,
            load_from_cache_file=True,
        )
