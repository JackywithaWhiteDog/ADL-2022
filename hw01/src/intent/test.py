from argparse import Namespace
import csv

import torch
from pytorch_lightning import Trainer

from src import logger
from src.intent.data_manager import IntentDataManager
from src.intent.models import IntentClassifier

def test(args: Namespace) -> None:
    data_manager = IntentDataManager(
        cache_dir=args.cache_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_file=args.test_file
    )

    test_dataloader = data_manager.get_test_dataloader()

    model = IntentClassifier.load_from_checkpoint(args.ckpt_path)

    if args.device.type == "cpu":
        trainer = Trainer(accelerator="cpu", deterministic=True)
    else:
        trainer = Trainer(
            devices=[args.device.index] if args.device.index else 1,
            accelerator="gpu",
            deterministic=True,
        )

    pred = trainer.predict(model, test_dataloader, return_predictions=True)
    pred = torch.cat(pred)
    result = [
        {
            "id": i,
            "intent": data_manager.idx2intent[p]
        }
        for i, p in zip(test_dataloader.dataset.ids, pred.tolist())
    ]
    with open(args.pred_file, "w") as csvfile:
        fieldnames = ["id", "intent"]
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        writer.writerows(result)
    logger.info(f"Prediction saved at {str(args.pred_file.resolve())}")
