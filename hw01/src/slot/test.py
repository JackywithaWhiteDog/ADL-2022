from argparse import Namespace
import csv

import torch
from pytorch_lightning import Trainer

from src import logger
from src.slot.data_manager import SlotDataManager
from src.slot.models import SlotTagger

def test(args: Namespace) -> None:
    data_manager = SlotDataManager(
        cache_dir=args.cache_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_file=args.test_file
    )

    test_dataloader = data_manager.get_test_dataloader()

    model = SlotTagger.load_from_checkpoint(args.ckpt_path)

    if args.device.type == "cpu":
        trainer = Trainer(accelerator="cpu", deterministic=True, logger=False)
    else:
        trainer = Trainer(
            devices=[args.device.index] if args.device.index else 1,
            accelerator="gpu",
            deterministic=True,
            logger=False,
        )

    pred = trainer.predict(model, test_dataloader, return_predictions=True)
    pred = [sen for batch in pred for sen in batch]
    result = [
        {
            'id': i,
            'tags': ' '.join([data_manager.idx2tag[idx] for idx in p.tolist()])
        }
        for i, p in zip(test_dataloader.dataset.ids, pred)
    ]
    with open(args.pred_file, "w") as csvfile:
        fieldnames = ["id", "tags"]
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        writer.writerows(result)
    logger.info(f"Prediction saved at {str(args.pred_file.resolve())}")
