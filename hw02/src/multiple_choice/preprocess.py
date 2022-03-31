from itertools import chain

from src.config import QUESTION_NAME, PARAGRAPHS_NAME

def preprocess_function(examples, tokenizer, max_seq_length, pad_to_max_length):
    first_sentences = [[question] * 4 for question in examples[QUESTION_NAME]]
    second_sentences = examples[PARAGRAPHS_NAME]

    # Flatten out
    first_sentences = list(chain.from_iterable(first_sentences))
    second_sentences = list(chain.from_iterable(second_sentences))

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length" if pad_to_max_length else False,
    )
    # Un-flatten
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
