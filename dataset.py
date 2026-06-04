"""
dataset.py — MS-COCO captions dataset + transforms + custom collate.

The collate_fn sorts the batch by caption length in **descending** order,
which is mandatory for torch.nn.utils.rnn.pack_padded_sequence.
"""

import json
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from vocabulary import Vocabulary


class CocoDataset(Dataset):
    """
    Each sample: one (image, caption) pair.
    MS-COCO has ~5 captions per image → ~413 K training samples total.
    """

    def __init__(
        self,
        img_dir:   str,
        ann_file:  str,
        vocab:     Vocabulary,
        transform=None,
        max_samples: int = -1,      # -1 = use all
    ):
        self.img_dir   = img_dir
        self.vocab     = vocab
        self.transform = transform

        with open(ann_file, "r") as f:
            data = json.load(f)

        # image_id → filename map
        id2fn = {img["id"]: img["file_name"] for img in data["images"]}

        self.samples: list[tuple[str, str]] = [
            (os.path.join(img_dir, id2fn[ann["image_id"]]), ann["caption"])
            for ann in data["annotations"]
            if ann["image_id"] in id2fn          # guard against missing files
        ]

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    # ── Dataset API ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, caption = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Encode to [<start>, w1, …, wN, <end>]
        ids    = self.vocab.encode(caption)
        target = torch.tensor(ids, dtype=torch.long)

        return image, target

    # ── Collate ──────────────────────────────────────────────────────────
    @staticmethod
    def collate_fn(batch, pad_idx: int):
        """
        1. Sort by caption length descending  (required for pack_padded_sequence).
        2. Pad captions to the longest in the batch.

        Returns
        -------
        images   : (B, C, H, W)
        captions : (B, T_max)   — includes <start> and <end>
        lengths  : (B,)         — actual caption lengths (with special tokens)
        """
        # Sort descending by caption length
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*batch)

        images  = torch.stack(images, 0)                         # (B, C, H, W)
        lengths = torch.tensor([len(cap) for cap in captions])  # (B,)

        max_len = lengths.max().item()
        padded  = torch.full((len(captions), max_len), pad_idx, dtype=torch.long)
        for i, cap in enumerate(captions):
            padded[i, : len(cap)] = cap

        return images, padded, lengths


# ── Transforms ────────────────────────────────────────────────────────────
def get_transform(
    train: bool,
    img_size:  int   = 256,
    crop_size: int   = 224,
    mean: tuple      = (0.485, 0.456, 0.406),
    std:  tuple      = (0.229, 0.224, 0.225),
):
    """
    Train:  Resize → RandomCrop(224) → RandomHorizontalFlip → Normalize
    Val:    Resize → CenterCrop(224)                         → Normalize
    """
    if train:
        return T.Compose([
            T.Resize(img_size),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize(img_size),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
