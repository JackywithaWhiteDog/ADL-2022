from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional

DEFAULT_DATA_DIR = Path("./data")

@dataclass
class MultiArguments:

    pretrained_model: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    context_path: Path = field(
        default=DEFAULT_DATA_DIR / "context.json",
        metadata={"help": "Path to context file"}
    )

    train_path: Path = field(
        default=DEFAULT_DATA_DIR / "train.json",
        metadata={"help": "Path to training dataset"}
    )

    valid_path: Path = field(
        default=DEFAULT_DATA_DIR / "valid.json",
        metadata={"help": "Path to validation dataset"}
    )

    cache_dir: Path = field(
        default=Path("./cache"),
        metadata={"help": "Directory for cache"}
    )

    output_dir: Path = field(
        default=Path("./ckpt"),
        metadata={"help": "Directory for output"}
    )

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Max sequence length"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for training dataset"}
    )
    valid_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for validation dataset"}
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "Steps for graient accumulation"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate"}
    )
    seed: int = field(
        default=1123,
        metadata={"help": "Random seed"}
    )

    que_col: str = field(
        default="question",
        metadata={"help": "Column name for question"}
    )

    para_col: str = field(
        default="paragraphs",
        metadata={"help": "Column name for paragraphs"}
    )

    num_para: int = field(
        default=4,
        metadata={"help": "Number of paragraphs"}
    )

    rel_col: str = field(
        default="relevant",
        metadata={"help": "Column name for relevant paragraph"}
    )

    label_col: str = field(
        default="labels",
        metadata={"help": "Column name for labels"}
    )
