from transformers import (
    HfArgumentParser,
    set_seed
)

from src.multiple_choice.train import train
from src.multiple_choice.args import MultiArguments

if __name__ == "__main__":
    parser = HfArgumentParser(MultiArguments)
    args = parser.parse_args_into_dataclasses()[0]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train(args)
