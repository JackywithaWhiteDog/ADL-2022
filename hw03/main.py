#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from pathlib import Path
import json
from typing import Dict, Any

import datasets
import nltk
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, load_metric, Metric, MetricInfo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    Adafactor,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.utils import get_full_repo_name, is_offline_mode

# https://github.com/JackywithaWhiteDog/ADL-2022/tree/main/hw03/tw_rouge
from tw_rouge import get_rouge

logger = logging.getLogger(__name__)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

class Rouge(Metric):
    def __init__(self, *args, **kwargs):
        super(Rouge, self).__init__(*args, **kwargs)

    def _info(self) -> MetricInfo:
        """Construct the MetricInfo object. See `MetricInfo` for details.
        Warning: This function is only called once and the result is cached for all
        following .info() calls.
        Returns:
            info: (MetricInfo) The metrics information
        """
        return MetricInfo(
            description="ROUGE score",
            citation="https://github.com/JackywithaWhiteDog/ADL-2022/tree/main/hw03/tw_rouge",
            features=datasets.Features(
                predictions=datasets.Value("string"),
                references=datasets.Value("string")
            )
        )

    def _compute(self, *args, predictions=None, references=None, **kwargs) -> Dict[str, Any]:
        """This method defines the common API for all the metrics in the library"""
        # print(f"predictions: {predictions}")
        # print(f"references: {references}")
        return get_rouge(predictions, references, ignore_empty=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    parser.add_argument("--adafactor", action="store_true", help="Use Adafactor, if False use AdamW")

    parser.add_argument("--do_train", action="store_true", help="Train the model")
    parser.add_argument("--do_predict", action="store_true", help="Predict on the testing dataset")
    parser.add_argument("--pred_file", type=str, default=None, help="Path to store prediction.")

    # Generation Strategies
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether or not to use sampling; use greedy decoding otherwise"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The value used to module the next token probabilities"
    )

    parser.add_argument("--rl", action="store_true", help="Fine tune model by reinforcement learning")
    parser.add_argument(
        "--rl_top_k",
        type=int,
        default=0,
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering"
    )
    parser.add_argument(
        "--rl_top_p",
        type=float,
        default=0.2,
        help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation"
    )
    parser.add_argument(
        "--rl_temperature",
        type=float,
        default=1.0,
        help="The value used to module the next token probabilities"
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None and args.test_file is None:
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv, json, or jsonl file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv, json, or jsonl file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`test_file` should be a csv, json, or jsonl file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else None
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        extension = "json"
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        if extension == "jsonl": # jsonlines file
            extension = "json"
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["test"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if args.do_train and summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    if args.do_predict:
        args.has_summary = summary_column in raw_datasets["test"].column_names

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        if summary_column in examples:
            targets = examples[summary_column]
        else:
            targets = [""] * len(examples[text_column])
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if args.do_predict:
        test_indices = raw_datasets["test"]["id"]
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.do_train:
        train_dataset = processed_datasets["train"]
        # train_dataset = train_dataset.select(range(10)) # For debuging
        eval_dataset = processed_datasets["validation"]
        # eval_dataset = eval_dataset.select(range(10)) # For debuging

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.do_predict:
        test_dataset = processed_datasets["test"]

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels=None):
        # rougeLSum expects newline after each sentence
        preds = [pred.strip() for pred in preds]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        if labels is None:
            return preds

        labels = [label.strip() for label in labels]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    if args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    if args.do_predict:
        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Metric
    metric = Rouge()
    rl_metric = CrossEntropyLoss(reduction='none')

    def generate_predictions(accelerator, model, tokenizer, batch, gen_kwargs, return_tokens=False):
        generated_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )

        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        if return_tokens:
            return decoded_preds, generated_tokens
        return decoded_preds


    # Train!
    if args.do_train:
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if args.adafactor:
            optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, relative_step=False)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Use reinforcement learning = {args.rl}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        logging_result = {
            'loss': [],
            'rouge': []
        }

        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        step_loss = 0
        for epoch in range(args.num_train_epochs):
            model.train()
            epoch_loss = 0
            epoch_steps = 0
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                if not args.rl:
                    loss = outputs.loss
                else:
                    with torch.no_grad():
                        gen_kwargs_greedy = {
                            "max_length": args.val_max_target_length if args is not None else config.max_length,
                            "do_sample": False,
                            "num_beams": 1,
                            "top_k": 0,
                            "top_p": 1,
                            "temperature": 1,
                        }
                        decoded_preds_greedy = generate_predictions(accelerator, model, tokenizer, batch, gen_kwargs_greedy)

                        gen_kwargs_sample = {
                            "max_length": args.val_max_target_length if args is not None else config.max_length,
                            "do_sample": True,
                            "num_beams": 1,
                            "top_k": args.rl_top_k,
                            "top_p": args.rl_top_p,
                            "temperature": args.rl_temperature,
                        }
                        decoded_preds_sample, sample_tokens = generate_predictions(accelerator, model, tokenizer, batch, gen_kwargs_sample, return_tokens=True)

                        labels = batch["labels"]
                        if not args.pad_to_max_length:
                            # If we did not pad to max length, we need to pad the labels too
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                        labels = accelerator.gather(labels).cpu().numpy()

                        if args.ignore_pad_token_for_loss:
                            # Replace -100 in the labels as we can't decode them.
                            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                        decoded_preds_greedy, decoded_labels = postprocess_text(decoded_preds_greedy, decoded_labels)
                        decoded_preds_sample = postprocess_text(decoded_preds_sample)

                        indices = [
                            i
                            for i, (g, s) in enumerate(zip(decoded_preds_greedy, decoded_preds_sample))
                            if len(g) > 0 and len(s) > 0
                        ]

                        decoded_labels = [decoded_labels[i] for i in indices]
                        decoded_preds_greedy = [decoded_preds_greedy[i] for i in indices]
                        decoded_preds_sample = [decoded_preds_sample[i] for i in indices]

                        greedy_scores = get_rouge(decoded_preds_greedy, decoded_labels, avg=False, ignore_empty=False)
                        sample_scores = get_rouge(decoded_preds_sample, decoded_labels, avg=False, ignore_empty=False)

                        score_weights = {
                            'rouge-1': {'f': 1.0 / 0.271},
                            'rouge-2': {'f': 1.0 / 0.102},
                            'rouge-l': {'f': 1.0 / 0.241}
                        }
                        num_scores = len([
                            0
                            for rouge, critic in score_weights.items()
                            for critic_name, weight in critic.items()
                        ])
                        greedy_rewards = torch.Tensor([
                            sum(
                                score[rouge][critic_name] * weight
                                for rouge, critic in score_weights.items()
                                for critic_name, weight in critic.items()
                            ) / num_scores
                            for score in greedy_scores
                        ])
                        sample_rewards = torch.Tensor([
                            sum(
                                score[rouge][critic_name] * weight
                                for rouge, critic in score_weights.items()
                                for critic_name, weight in critic.items()
                            ) / num_scores
                            for score in sample_scores
                        ])

                    logits = outputs.logits
                    length = min(sample_tokens.shape[1], logits.shape[1])
                    logits = logits[indices, :length, :].reshape(-1, logits.shape[-1])
                    sample_tokens = torch.Tensor(sample_tokens)[indices, :length].reshape(-1).long().to(logits.device)
                    ce_loss = rl_metric(logits, sample_tokens)
                    rewards = (sample_rewards - greedy_rewards).reshape(-1, 1).to(ce_loss.device)
                    ce_loss = ce_loss.reshape(rewards.shape[0], -1)
                    loss = (ce_loss * rewards).mean()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                step_loss += loss.item()
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    epoch_loss = epoch_loss * epoch_steps + step_loss
                    epoch_steps += 1
                    epoch_loss /= epoch_steps
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(f"Epoch [{epoch+1}/{args.num_train_epochs}] Total Loss: {epoch_loss:.3f} (Current Step: {step_loss:.3f})")
                    completed_steps += 1
                    logging_result['loss'].append(step_loss)
                    step_loss = 0

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            if args.val_max_target_length is None:
                args.val_max_target_length = args.max_target_length

            gen_kwargs = {
                "max_length": args.val_max_target_length if args is not None else config.max_length,
                "do_sample": args.do_sample,
                "num_beams": args.num_beams,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
            }
            for step, batch in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_local_main_process, leave=True)):
                with torch.no_grad():
                    decoded_preds = generate_predictions(accelerator, model, tokenizer, batch, gen_kwargs)

                    labels = batch["labels"]
                    if not args.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                    labels = accelerator.gather(labels).cpu().numpy()

                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                    metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            result = metric.compute()

            logging_result['rouge'].append(result)

            # Extract f1 score from result
            result = {key: value["f"] * 100 for key, value in result.items()}

            result = {k: round(v, 4) for k, v in result.items()}

            logger.info(result)

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
                with open(f"{args.output_dir}/results.json", "w") as f:
                    json.dump(logging_result, f, indent=2)

    if args.do_predict:
        # Prepare everything with our `accelerator`.
        model, test_dataloader = accelerator.prepare(
            model, test_dataloader
        )

        model.eval()
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        }
        pred = []
        for step, batch in enumerate(tqdm(test_dataloader, disable=not accelerator.is_local_main_process, leave=True, desc="Do prediction")):
            with torch.no_grad():
                decoded_preds = generate_predictions(accelerator, model, tokenizer, batch, gen_kwargs)

                if args.has_summary:
                    labels = batch["labels"]
                    if not args.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                    labels = accelerator.gather(labels).cpu().numpy()

                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                    metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                else:
                    decoded_preds = postprocess_text(decoded_preds)
                pred.extend(decoded_preds)
        if args.has_summary:
            result = metric.compute()
            logger.info(result)
        args.pred_file = Path(args.pred_file)
        args.pred_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.pred_file, "w") as f:
            for idx, title in zip(test_indices, pred):
                json.dump({
                    "title": title,
                    "id": idx
                }, f)
                f.write('\n')
        logger.info("Completed")

if __name__ == "__main__":
    main()
