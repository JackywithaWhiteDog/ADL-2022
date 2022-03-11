import json
from collections import Counter
from pathlib import Path

from src import logger
from src.utils.preprocess import build_index, build_vocab, build_embedding, get_glove

MARKS = ".,;!?()"

def filter_marks(text: str) -> str:
    result = text
    for mark in MARKS:
        result = result.replace(mark, " ")
    return result

def preprocess(
    data_dir: Path,
    vocab_size: int,
    glove_path: Path,
    output_dir: Path,
) -> None:
    intents = set()
    words = Counter()
    for split in ("train", "eval"):
        dataset_path = data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logger.info(f"Dataset loaded from {str(dataset_path.resolve())}")

        intents.update({instance["intent"] for instance in dataset})
        words.update([
            token.lower()
            for instance in dataset
            for token in filter_marks(instance["text"]).split()
        ])
    if vocab_size <= 0:
        vocab_size = None
    common_words = {w for w, _ in words.most_common(vocab_size)}
    glove = get_glove(
        glove_path=glove_path,
        common_words=common_words
    )
    common_words = {
        word
        for word in common_words
        if word in glove
    }

    intent_idx_path = output_dir / "intent2idx.json"
    build_index(targets=intents, output_file=intent_idx_path)
    vocab = build_vocab(
        common_words=common_words,
        vocab_size=vocab_size,
        output_dir=output_dir,
    )
    build_embedding(
        vocab=vocab,
        common_words=common_words,
        vocab_size=vocab_size,
        glove_path=glove_path,
        output_dir=output_dir,
        glove=glove
    )
