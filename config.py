"""
config.py — Centralised hyperparameter and path configuration.
Edit this file to switch between variants (backbone, cell_type, epochs, etc.)
"""

from dataclasses import dataclass
from typing import Tuple
import os
import torch


@dataclass
class Config:
    # ──────────────────────────────────────────────────────────
    # Paths
    # ──────────────────────────────────────────────────────────
    coco_root:      str = "./data/coco"
    train_ann:      str = "./data/coco/annotations/captions_train2014.json"
    val_ann:        str = "./data/coco/annotations/captions_val2014.json"
    train_img_dir:  str = "./data/coco/train2014"
    val_img_dir:    str = "./data/coco/val2014"
    vocab_path:     str = "./data/vocab.pkl"
    checkpoint_dir: str = "./checkpoints"
    results_dir:    str = "./results"
    val_ext_dir:    str = "./imagenes_validation"   # ≥10 external images here

    # ──────────────────────────────────────────────────────────
    # Vocabulary
    # ──────────────────────────────────────────────────────────
    vocab_threshold: int = 4            # discard words with freq < threshold

    # ──────────────────────────────────────────────────────────
    # Model architecture
    # ──────────────────────────────────────────────────────────
    embed_size:       int   = 256
    hidden_size:      int   = 512
    num_layers:       int   = 1
    dropout:          float = 0.5
    encoder_backbone: str   = "resnet50"   # "resnet50" | "resnet101"
    cell_type:        str   = "lstm"       # "lstm" | "gru"  (variant)

    # ──────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────
    batch_size:        int   = 32
    num_epochs:        int   = 10
    learning_rate:     float = 1e-3
    finetune_epoch:    int   = 6       # unfreeze CNN backbone from this epoch
    finetune_lr:       float = 1e-4    # lower LR for fine-tuning
    log_step:          int   = 100     # log every N mini-batches
    max_train_samples: int   = -1      # -1 = full dataset; set small for debug

    # ──────────────────────────────────────────────────────────
    # Image preprocessing
    # ──────────────────────────────────────────────────────────
    img_size:      int   = 256
    crop_size:     int   = 224
    imagenet_mean: Tuple = (0.485, 0.456, 0.406)
    imagenet_std:  Tuple = (0.229, 0.224, 0.225)

    # ──────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────
    max_seq_len: int = 25
    beam_size:   int = 3

    # ──────────────────────────────────────────────────────────
    # Hardware  (RTX 5060 Ti / Blackwell / cu128)
    # ──────────────────────────────────────────────────────────
    device:      str  = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int  = 6        # adjust to your CPU core count
    pin_memory:  bool = True
    use_amp:     bool = True     # mixed-precision — essential for Blackwell

    # ──────────────────────────────────────────────────────────
    def __post_init__(self):
        for d in [self.checkpoint_dir, self.results_dir,
                  self.val_ext_dir, "./data"]:
            os.makedirs(d, exist_ok=True)
