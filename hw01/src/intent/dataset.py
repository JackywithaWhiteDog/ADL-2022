from typing import List, Optional, Union, Tuple

from torch.utils import data
from torch import Tensor

class IntentDataset(data.Dataset):
    def __init__(self, x: Tensor, ids: List[str], length: Tensor, y: Optional[Tensor]=None) -> None:
        self.data = x
        self.ids = ids
        self.length = length
        self.label = y

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        if self.label is None:
            return self.data[idx], self.length[idx]
        return self.data[idx], self.length[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
