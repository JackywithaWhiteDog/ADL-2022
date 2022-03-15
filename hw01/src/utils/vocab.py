from typing import Iterable, List

def pad_to_len(seqs: List[List[int]], max_len: int, padding: int) -> List[List[int]]:
    max_seq_len = max(len(seq) for seq in seqs)
    to_len = max_seq_len if max_len < 0 else min(max_len, max_seq_len)
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds

class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            self.PAD: 0,
            self.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], max_len: int = -1
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        padded_ids = pad_to_len(batch_ids, max_len, self.pad_id)
        return padded_ids
