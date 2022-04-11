import csv
from functools import partial
from pathlib import Path

import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from src import get_logger

from src.multiple_choice.datasets import get_datasets as get_multi_datasets
from src.multiple_choice.data_collator import DataCollatorForMultipleChoice
from src.multiple_choice.preprocess import preprocess_function as multi_preprocess_function
from src.multiple_choice.metrics import compute_accuracy

from src.qa.datasets import get_datasets as get_qa_datasets
from src.qa.preprocess import prepare_validation_features as qa_prepare_validation_features
from src.qa.trainer import QuestionAnsweringTrainer
from src.qa.postprocess import post_processing_function
from src.qa.metrics import compute_metrics

from src.predict.args import ModelArguments, DataArguments

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert data_args.output_file is not None
    data_args.output_file = Path(data_args.output_file)
    data_args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_level = training_args.get_process_log_level()
    logger = get_logger(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    assert data_args.test_file is not None
    data_files = {}
    data_files["test"] = data_args.test_file
    raw_datasets = get_multi_datasets(data_files, data_args.context_file)

    # Load pretrained model and tokenizer

    # download model & vocab.
    multi_config = AutoConfig.from_pretrained(
        model_args.multi_model,
        cache_dir=model_args.cache_dir,
    )
    multi_tokenizer = AutoTokenizer.from_pretrained(
        model_args.multi_model,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    multi_model = AutoModelForMultipleChoice.from_pretrained(
        model_args.multi_model,
        from_tf=bool(".ckpt" in model_args.multi_model),
        config=multi_config,
        cache_dir=model_args.cache_dir,
    )

    if data_args.multi_max_seq_length is None:
        multi_max_seq_length = multi_tokenizer.model_max_length
        if multi_max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({multi_tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            multi_max_seq_length = 1024
    else:
        if data_args.multi_max_seq_length > multi_tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.multi_max_seq_length}) is larger than the maximum length for the"
                f"model ({multi_tokenizer.model_max_length}). Using max_seq_length={multi_tokenizer.model_max_length}."
            )
        multi_max_seq_length = min(data_args.multi_max_seq_length, multi_tokenizer.model_max_length)

    test_dataset = raw_datasets["test"]
    with training_args.main_process_first(desc="testing dataset map pre-processing"):
        test_dataset = test_dataset.map(
            partial(
                multi_preprocess_function,
                tokenizer=multi_tokenizer,
                max_seq_length=multi_max_seq_length,
                pad_to_max_length=data_args.pad_to_max_length
            ),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    multi_data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=multi_tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Initialize our Trainer
    multi_trainer = Trainer(
        model=multi_model,
        args=training_args,
        tokenizer=multi_tokenizer,
        data_collator=multi_data_collator,
        compute_metrics=compute_accuracy,
    )

    multi_pred = multi_trainer.predict(test_dataset=test_dataset)
    relevant_ids = multi_pred.predictions.argmax(axis=1)
    if multi_pred.label_ids is not None:
        logger.info(f"Multiple Choice Accuracy: {(relevant_ids == multi_pred.label_ids).mean():.2f}")
    else:
        logger.info(relevant_ids)

    raw_datasets = get_qa_datasets(data_files, data_args.context_file, relevant_ids)

    # download model & vocab.
    qa_config = AutoConfig.from_pretrained(
        model_args.qa_model,
        cache_dir=model_args.cache_dir,
    )
    qa_tokenizer = AutoTokenizer.from_pretrained(
        model_args.qa_model,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.qa_model,
        from_tf=bool(".ckpt" in model_args.qa_model),
        config=qa_config,
        cache_dir=model_args.cache_dir,
    )

    column_names = raw_datasets["test"].column_names

    if data_args.qa_max_seq_length > qa_tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.qa_max_seq_length}) is larger than the maximum length for the"
            f"model ({qa_tokenizer.model_max_length}). Using max_seq_length={qa_tokenizer.model_max_length}."
        )
    qa_max_seq_length = min(data_args.qa_max_seq_length, qa_tokenizer.model_max_length)
    test_examples = raw_datasets["test"]
    with training_args.main_process_first(desc="testing dataset map pre-processing"):
        test_dataset = test_examples.map(
            partial(
                qa_prepare_validation_features,
                tokenizer=qa_tokenizer,
                max_seq_length=qa_max_seq_length,
                doc_stride=data_args.doc_stride,
                pad_to_max_length=data_args.pad_to_max_length
            ),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    qa_data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(qa_tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    qa_trainer = QuestionAnsweringTrainer(
        model=qa_model,
        args=training_args,
        tokenizer=qa_tokenizer,
        data_collator=qa_data_collator,
        post_process_function=partial(
            post_processing_function,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=None,
            log_level=log_level
        ),
        compute_metrics=partial(compute_metrics, version_2_with_negative=data_args.version_2_with_negative),
    )

    qa_pred = qa_trainer.predict(predict_dataset=test_dataset, predict_examples=test_examples)
    if qa_pred.label_ids is not None:
        logger.info(f"QA Accuracy: {sum(pred['prediction_text'] == label['answers']['text'][0] for pred, label in zip(qa_pred.predictions, qa_pred.label_ids)) / len(test_examples):.2f}")

    result = [
        {
            "id": pred["id"],
            "answer": pred["prediction_text"]
        }
        for pred in qa_pred.predictions
    ]
    with open(data_args.output_file, "w") as csvfile:
        fieldnames = ["id", "answer"]
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        writer.writerows(result)
    logger.info(f"Prediction saved at {str(data_args.output_file)}")

if __name__ == "__main__":
    main()