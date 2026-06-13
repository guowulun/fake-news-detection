import os
import numpy as np
from collections import Counter


def build_vocab(texts, min_freq=2, max_vocab=50000):
    """
    Build word → index vocab from a list of cleaned texts.
    Index 0 = PAD, index 1 = UNK.
    """
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    print(f"[Vocab] Size: {len(vocab):,} (min_freq={min_freq})")
    return vocab


def load_glove(glove_path, vocab, embed_dim=300):
    """
    Load GloVe vectors for words in vocab.
    Returns embedding matrix of shape (vocab_size, embed_dim).
    Download: https://nlp.stanford.edu/data/glove.6B.zip
    """
    vocab_size = len(vocab)
    matrix = np.random.normal(0, 0.1, (vocab_size, embed_dim)).astype(np.float32)
    matrix[0] = 0  # PAD = zeros

    if not os.path.exists(glove_path):
        print(f"[GloVe] File not found: {glove_path}. Using random init.")
        return matrix

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                idx = vocab[word]
                matrix[idx] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"[GloVe] Loaded {found}/{vocab_size} vectors from {glove_path}")
    return matrix
