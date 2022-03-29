from dataclasses import dataclass, field
from typing import Union, Optional
from itertools import chain

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils import PaddingStrategy

@dataclass
class DataCollator:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    para_col: str = "paragraphs"
    label_col: str = "labels"

    def __call__(self, features):
        labels = [feature.pop(self.label_col) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        # Flatten
        flattened_features  = [
            [
                {k: v[i] for k, v in feature.items()}
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain.from_iterable(flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch[self.label_col] = torch.tensor(labels, dtype=torch.int64)
        return batch