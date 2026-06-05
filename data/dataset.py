"""
Data loading & preprocessing for LIAR and FakeNewsNet datasets.

LIAR: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
FakeNewsNet: https://github.com/KaiDMML/FakeNewsNet
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch


# ─────────────────────────────────────────────
# Label mappings
# ─────────────────────────────────────────────

LIAR_BINARY_MAP = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0,
}

FAKENEWSNET_BINARY_MAP = {
    "real": 1,
    "fake": 0,
}


# ─────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # keep alphanum + spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# LIAR loader
# LIAR TSV columns (no header):
# 0:id 1:label 2:statement 3:subject 4:speaker 5:job 6:state
# 7:party 8:barely_true 9:false 10:half_true 11:mostly_true
# 12:pants_fire 13:context
# ─────────────────────────────────────────────

def load_liar(data_dir: str) -> pd.DataFrame:
    """
    Loads LIAR train/valid/test TSV files and returns a combined DataFrame.
    Expected files: train.tsv, valid.tsv, test.tsv inside data_dir.
    """
    cols = ["id", "label", "statement", "subject", "speaker",
            "job", "state", "party", "barely_true", "false_count",
            "half_true", "mostly_true", "pants_fire", "context"]

    dfs = []
    for split in ["train", "valid", "test"]:
        path = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(path):
            print(f"[LIAR] Warning: {path} not found, skipping.")
            continue
        df = pd.read_csv(path, sep="\t", header=None, names=cols)
        df["split"] = split
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No LIAR TSV files found in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df["text"] = df["statement"].apply(clean_text)
    df["binary_label"] = df["label"].map(LIAR_BINARY_MAP)
    df = df.dropna(subset=["binary_label"])
    df["binary_label"] = df["binary_label"].astype(int)
    df["source"] = "liar"
    return df[["text", "binary_label", "split", "source"]]


# ─────────────────────────────────────────────
# FakeNewsNet loader
# Assumes CSV with columns: title (or text), label
# ─────────────────────────────────────────────

def load_fakenewsnet(data_dir: str, text_col: str = "title") -> pd.DataFrame:
    """
    Loads FakeNewsNet CSV files.
    Expects: politifact_fake.csv, politifact_real.csv,
             gossipcop_fake.csv, gossipcop_real.csv
    (or a combined fakenewsnet.csv with 'label' column)
    """
    records = []

    # Try combined file first
    combined = os.path.join(data_dir, "fakenewsnet.csv")
    if os.path.exists(combined):
        df = pd.read_csv(combined)
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not in {combined}. Available: {list(df.columns)}")
        df["text"] = df[text_col].apply(clean_text)
        df["binary_label"] = df["label"].str.lower().map(FAKENEWSNET_BINARY_MAP)
    else:
        # Try individual files
        file_map = {
            "politifact_fake.csv": "fake",
            "politifact_real.csv": "real",
            "gossipcop_fake.csv": "fake",
            "gossipcop_real.csv": "real",
        }
        for fname, lbl in file_map.items():
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                continue
            tmp = pd.read_csv(fpath)
            if text_col not in tmp.columns:
                continue
            tmp["text"] = tmp[text_col].apply(clean_text)
            tmp["binary_label"] = FAKENEWSNET_BINARY_MAP[lbl]
            records.append(tmp[["text", "binary_label"]])

        if not records:
            raise FileNotFoundError(f"No FakeNewsNet files found in {data_dir}")
        df = pd.concat(records, ignore_index=True)

    df = df.dropna(subset=["binary_label"])
    df["binary_label"] = df["binary_label"].astype(int)

    # Create splits
    train, temp = train_test_split(df, test_size=0.2, random_state=42,
                                   stratify=df["binary_label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42,
                                 stratify=temp["binary_label"])
    train["split"] = "train"
    val["split"] = "valid"
    test["split"] = "test"

    df = pd.concat([train, val, test], ignore_index=True)
    df["source"] = "fakenewsnet"
    return df[["text", "binary_label", "split", "source"]]


# ─────────────────────────────────────────────
# Combined loader
# ─────────────────────────────────────────────

def load_combined(liar_dir: str, fnn_dir: str) -> pd.DataFrame:
    """Load and merge both datasets."""
    dfs = []
    try:
        dfs.append(load_liar(liar_dir))
        print(f"[LIAR] Loaded successfully.")
    except Exception as e:
        print(f"[LIAR] Failed: {e}")

    try:
        dfs.append(load_fakenewsnet(fnn_dir))
        print(f"[FakeNewsNet] Loaded successfully.")
    except Exception as e:
        print(f"[FakeNewsNet] Failed: {e}")

    if not dfs:
        raise RuntimeError("No datasets could be loaded.")

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["text"].str.len() > 5]  # filter very short texts
    return df


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)


def print_stats(df: pd.DataFrame):
    print("\n── Dataset statistics ──")
    print(f"Total samples : {len(df):,}")
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        print(f"\n  [{src.upper()}] {len(sub):,} samples")
        for sp in ["train", "valid", "test"]:
            s = sub[sub["split"] == sp]
            if len(s):
                fake = (s["binary_label"] == 0).sum()
                real = (s["binary_label"] == 1).sum()
                print(f"    {sp:6s}: {len(s):5,}  (real={real}, fake={fake})")
    print()


# ─────────────────────────────────────────────
# PyTorch Dataset wrappers
# ─────────────────────────────────────────────

class LSTMDataset(Dataset):
    """Dataset for LSTM: returns padded token id sequences."""

    def __init__(self, texts, labels, vocab, max_len=128):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sequences = []
        for txt in texts:
            tokens = txt.split()[:max_len]
            ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens]
            # Pad / truncate
            ids = ids + [0] * (max_len - len(ids))
            self.sequences.append(ids)
        self.sequences = torch.tensor(self.sequences, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class BERTDataset(Dataset):
    """Dataset for BERT: returns tokenizer outputs."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
