from argparse import Namespace

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer
)

from src.utils.data_manager import DataManager
from src.utils.data_collator import DataCollator
from src.utils.metrics import accuracy

def train(args: Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        cache_dir=args.cache_dir,
        model_revision="main",
        use_fast=True
    )
    
    data_manager = DataManager(
        tokenizer=tokenizer,
        context_path=args.context_path,
        train_path=args.train_path,
        valid_path=args.valid_path,
        max_seq_length=args.max_seq_length,
        que_col=args.que_col,
        para_col=args.para_col,
        num_para=args.num_para,
        rel_col=args.rel_col,
        label_col=args.label_col
    )

    data_collator = DataCollator(
        tokenizer=tokenizer,
        para_col=args.para_col,
        label_col=args.label_col
    )

    config = AutoConfig.from_pretrained(
        args.pretrained_model,
        cache_dir=args.cache_dir,
        revision="main",
        use_auth_token=None
    )

    model = AutoModelForMultipleChoice.from_config(config)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.valid_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # no_cuda=True,
        ),
        train_dataset=data_manager.train_dataset,
        eval_dataset=data_manager.valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=accuracy,
    )

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    metrics["train_samples"] = len(data_manager.train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(data_manager.valid_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

