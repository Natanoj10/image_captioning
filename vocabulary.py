"""
vocabulary.py — Vocabulary mapping words ↔ integer indices.

Special tokens (fixed indices):
    <pad>   = 0   padding token
    <start> = 1   beginning of sequence
    <end>   = 2   end of sequence
    <unk>   = 3   out-of-vocabulary words

Usage:
    vocab = build_vocab(ann_file, threshold=4, save_path="data/vocab.pkl")
    vocab = Vocabulary.load("data/vocab.pkl")
    ids   = vocab.encode("a dog sitting on a bench")
    text  = vocab.decode([5, 12, 3])
"""

import json
import pickle
from collections import Counter

import nltk

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


class Vocabulary:
    """Bidirectional word ↔ index mapping with special tokens."""

    PAD_TOKEN   = "<pad>"    # idx = 0
    START_TOKEN = "<start>"  # idx = 1
    END_TOKEN   = "<end>"    # idx = 2
    UNK_TOKEN   = "<unk>"    # idx = 3

    _SPECIAL = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._next = 0
        for tok in self._SPECIAL:
            self._add(tok)

    # ── Internal ─────────────────────────────────────────────────────────
    def _add(self, word: str) -> None:
        if word not in self.word2idx:
            self.word2idx[word] = self._next
            self.idx2word[self._next] = word
            self._next += 1

    # ── Build ────────────────────────────────────────────────────────────
    def build_from_captions(self, captions: list[str], threshold: int = 4) -> None:
        """Add words with frequency ≥ threshold to the vocabulary."""
        counter: Counter = Counter()
        for cap in captions:
            tokens = nltk.tokenize.word_tokenize(cap.lower())
            counter.update(tokens)

        added = 0
        for word, freq in counter.most_common():
            if freq >= threshold:
                self._add(word)
                added += 1

        print(f"[Vocab] {len(self):,} words  "
              f"(added {added:,} from corpus; threshold={threshold})")

    # ── Encode / Decode ──────────────────────────────────────────────────
    def encode(self, caption: str) -> list[int]:
        """String → [<start>, w1, w2, …, <end>]"""
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        ids = [self.start_idx]
        ids += [self.word2idx.get(t, self.unk_idx) for t in tokens]
        ids += [self.end_idx]
        return ids

    def decode(self, indices: list[int]) -> str:
        """Index list → human-readable string (strips special tokens)."""
        skip = {self.pad_idx, self.start_idx, self.end_idx}
        words = []
        for idx in indices:
            if idx == self.end_idx:
                break
            if idx not in skip:
                words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return " ".join(words)

    # ── Properties ───────────────────────────────────────────────────────
    @property
    def pad_idx(self)   -> int: return self.word2idx[self.PAD_TOKEN]
    @property
    def start_idx(self) -> int: return self.word2idx[self.START_TOKEN]
    @property
    def end_idx(self)   -> int: return self.word2idx[self.END_TOKEN]
    @property
    def unk_idx(self)   -> int: return self.word2idx[self.UNK_TOKEN]

    def __call__(self, word: str) -> int:
        return self.word2idx.get(word, self.unk_idx)

    def __len__(self) -> int:
        return len(self.word2idx)

    # ── Serialisation ────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Vocab] Saved → {path}")

    @staticmethod
    def load(path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Standalone builder ────────────────────────────────────────────────────
def build_vocab(ann_file: str, threshold: int, save_path: str) -> "Vocabulary":
    """Build vocabulary from a COCO captions annotation file."""
    with open(ann_file, "r") as f:
        data = json.load(f)
    captions = [ann["caption"] for ann in data["annotations"]]
    vocab = Vocabulary()
    vocab.build_from_captions(captions, threshold)
    vocab.save(save_path)
    return vocab
