from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
from pytorch_lightning import seed_everything

from src.slot.train import train

def main(args: Namespace) -> None:
    seed_everything(args.rand_seed)
    train(args)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )

    timezone_offset = 8
    tzinfo = timezone(timedelta(hours=timezone_offset))
    current_time = datetime.now(tzinfo).strftime("%Y%m%d_%H%M")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default=f"./ckpt/slot/{current_time}/",
    )

    parser.add_argument("--rand_seed", type=int, default=1123)

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--bidirectional", action="store_true", default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--train_with_eval", action="store_true")

    # Fine tuning
    parser.add_argument("--tune", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if not args.tune:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
