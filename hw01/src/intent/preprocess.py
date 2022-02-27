import json
from collections import Counter
from pathlib import Path
import pickle
from typing import List
from random import random
import re

from tqdm.auto import tqdm
import torch

from src import logger
from src.utils import filter_marks
from src.utils.vocab import Vocab

def build_index(
    intents: set,
    output_dir: Path
) -> None:
    intent2idx = {tag: i for i, tag in enumerate(intents)}
    intent_tag_path = output_dir / "intent2idx.json"
    intent_tag_path.write_text(json.dumps(intent2idx, indent=2))
    logger.info(f"Intent-2-Index saved at {str(intent_tag_path.resolve())}")

def build_vocab(
    common_words: set,
    vocab_size: int,
    output_dir: Path
) -> Vocab:
    vocab = Vocab(common_words)
    vocab_path = output_dir / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logger.info(f"Vocab saved at {str(vocab_path.resolve())}")
    return vocab

def build_embedding(
    vocab: Vocab,
    common_words: set,
    vocab_size: int,
    glove_path: Path,
    output_dir: Path
) -> None:
    glove: Dict[str, List[float]] = {}
    logger.info(f"Loading glove: {str(glove_path.resolve())}")
    with open(glove_path, "r") as f:
        row1 = f.readline()
        # if the first row is not header
        if not re.match("^[0-9]+ [0-9]+$", row1):
            # seek to 0
            f.seek(0)
        # otherwise ignore the header

        for i, line in enumerate(tqdm(f)):
            cols = line.rstrip().split(" ")
            word = cols[0]
            # skip word not in words if words are provided
            if word not in common_words:
                continue

            vector = [float(v) for v in cols[1:]]
            glove[word] = vector
            glove_dim = len(vector)

    assert len(glove) > 0
    assert vocab_size is None or len(glove) <= vocab_size
    assert all(len(v) == glove_dim for v in glove.values())

    num_tokens = len(vocab.tokens)
    num_matched = sum([token in glove for token in vocab.tokens])
    logger.info(
        f"Token covered: {num_matched} / {num_tokens} = {num_matched / num_tokens}"
    )

    embeddings: List[List[float]] = [
        glove.get(token, [random() * 2 - 1 for _ in range(glove_dim)])
        for token in vocab.tokens
    ]
    embeddings = torch.tensor(embeddings)
    embeddings_path = output_dir / "embeddings.pt"
    torch.save(embeddings, str(embeddings_path))
    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Embedding saved at {str(embeddings_path.resolve())}")

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

    build_index(intents=intents, output_dir=output_dir)
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
        output_dir=output_dir
    )
