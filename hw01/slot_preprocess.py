from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import random

from pytorch_lightning import seed_everything

from src import logger
from src.slot.preprocess import preprocess

def main(args: Namespace) -> None:
    seed_everything(args.rand_seed)

    preprocess(
        data_dir=args.data_dir,
        vocab_size=args.vocab_size,
        glove_path=args.glove_path,
        output_dir=args.output_dir
    )

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="Path to Glove Embedding.",
        default="./glove.840B.300d.txt",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        help="Random seed.",
        default=13,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Number of token in the vocabulary",
        default=-1,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
