from argparse import Namespace

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src import logger
from src.utils.early_stopping import EarlyStoppingWarmup
from src.intent.data_manager import IntentDataManager
from src.intent.models import IntentClassifier

def train(args: Namespace) -> None:
    data_manager = IntentDataManager(
        cache_dir=args.cache_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    train_dataloader = data_manager.get_train_dataloader()
    valid_dataloader = data_manager.get_valid_dataloader()

    model = IntentClassifier(
        embeddings=data_manager.embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=data_manager.num_class,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Train the model
    early_stopping = EarlyStoppingWarmup(
        warmup=10,
        monitor="val_acc",
        mode="max",
        min_delta=0,
        patience=5,
        check_on_train_epoch_end=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        dirpath=args.ckpt_dir,
        filename="intent-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}",
        save_top_k=1,
        save_on_train_epoch_end=False
    )

    if args.device.type == "cpu":
        trainer = Trainer(
            accelerator="cpu",
            deterministic=True,
            max_epochs=args.num_epoch,
            callbacks=[early_stopping, checkpoint_callback],
            gradient_clip_val=1,
            auto_lr_find=True,
            # profiler="simple"
        )
    else:
        trainer = Trainer(
            devices=[args.device.index] if args.device.index else 1,
            accelerator="gpu",
            deterministic=True,
            max_epochs=args.num_epoch,
            callbacks=[early_stopping, checkpoint_callback],
            gradient_clip_val=1,
            auto_lr_find=True,
            # profiler="simple"
        )

    logger.info("Training Start")
    if args.tune:
        result = trainer.tune(model, train_dataloader, valid_dataloader)
    else:
        trainer.fit(model, train_dataloader, valid_dataloader)
        logger.info(f"Best model saved at {checkpoint_callback.best_model_path}")
