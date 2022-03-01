import json
from pathlib import Path
import pickle
from random import random
import re
from typing import List, Dict

import torch
from tqdm.auto import tqdm

from src import logger
from src.utils.vocab import Vocab

def build_index(
    targets: set,
    output_file: Path
) -> None:
    target2idx = {target: idx for idx, target in enumerate(targets)}
    output_file.write_text(json.dumps(target2idx, indent=2))
    logger.info(f"Index saved at {str(output_file.resolve())}")

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
