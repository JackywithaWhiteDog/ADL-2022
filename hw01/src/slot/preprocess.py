import json
from collections import Counter
from pathlib import Path

from src import logger
from src.utils.preprocess import build_index, build_vocab, build_embedding, get_glove

def preprocess(
    data_dir: Path,
    vocab_size: int,
    glove_path: Path,
    output_dir: Path,
) -> None:
    tags = set()
    words = Counter()
    for split in ("train", "eval"):
        dataset_path = data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logger.info(f"Dataset loaded from {str(dataset_path.resolve())}")

        tags.update({
            tag
            for instance in dataset
            for tag in instance["tags"]
        })
        words.update([
            token.lower()
            for instance in dataset
            for token in instance["tokens"]
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

    tag_idx_path = output_dir / "tag2idx.json"
    build_index(targets=tags, output_file=tag_idx_path)
    vocab = build_vocab(
        common_words=common_words,
        vocab_size=vocab_size,
        output_dir=output_dir
    )
    build_embedding(
        vocab=vocab,
        common_words=common_words,
        vocab_size=vocab_size,
        glove_path=glove_path,
        output_dir=output_dir,
        glove=glove
    )
