from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from pytorch_lightning import seed_everything

from src.intent.test import test

def main(args: Namespace) -> None:
    seed_everything(args.rand_seed)
    test(args)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    parser.add_argument("--rand_seed", type=int, default=1123)

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    main(args)