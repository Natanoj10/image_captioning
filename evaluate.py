"""
evaluate.py — BLEU-1 / BLEU-4 evaluation on the MS-COCO validation set.

Compares:
  - Greedy decoding  (fast baseline)
  - Beam Search      (quality variant)

Each image has up to 5 reference captions in COCO; all are passed to
corpus_bleu so the metric accounts for multi-reference paraphrasing.

Usage
-----
# Single checkpoint (greedy + beam):
    python evaluate.py --ckpt checkpoints/best_model.ckpt

# Compare fine-tuned vs frozen base:
    python evaluate.py \
        --ckpt      checkpoints/best_model.ckpt \
        --base_ckpt checkpoints/base_model.ckpt

# Quick smoke-test on 500 images:
    python evaluate.py --ckpt checkpoints/best_model.ckpt --max_images 500
"""

import argparse
import json
import os
from collections import defaultdict

import nltk
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image
from tqdm import tqdm

from config import Config
from dataset import get_transform
from models import DecoderRNN, EncoderCNN
from vocabulary import Vocabulary

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config reconstruction helper
# ─────────────────────────────────────────────────────────────────────────────

def _as_cfg(raw) -> Config:
    """Accept either a Config dataclass or the legacy cfg.__dict__ (plain dict)."""
    if isinstance(raw, Config):
        return raw
    # Reconstruct from dict, ignoring keys that no longer exist in the dataclass
    valid_keys = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return Config(**{k: v for k, v in raw.items() if k in valid_keys})


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, vocab: Vocabulary, device: str):
    """Restore encoder + decoder from a .ckpt file. Returns (enc, dec, cfg)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = _as_cfg(ckpt["cfg"])

    enc = EncoderCNN(mcfg.embed_size, mcfg.encoder_backbone).to(device)
    dec = DecoderRNN(
        mcfg.embed_size, mcfg.hidden_size, len(vocab),
        mcfg.num_layers, mcfg.dropout, mcfg.cell_type,
    ).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval()
    dec.eval()
    return enc, dec, mcfg


# ─────────────────────────────────────────────────────────────────────────────
# Multi-reference dict from COCO annotation file
# ─────────────────────────────────────────────────────────────────────────────

def build_references(ann_file: str) -> dict[int, list[list[str]]]:
    """
    Returns {image_id: [[tok, tok, …], [tok, tok, …], …]}
    Each inner list is one tokenised reference caption (≤5 per image).
    """
    with open(ann_file) as f:
        data = json.load(f)

    refs: dict[int, list[list[str]]] = defaultdict(list)
    for ann in data["annotations"]:
        tokens = nltk.tokenize.word_tokenize(ann["caption"].lower())
        refs[ann["image_id"]].append(tokens)
    return refs


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    encoder,
    decoder,
    vocab: Vocabulary,
    ann_file: str,
    img_dir: str,
    cfg: Config,
    beam_size: int = 3,
    max_images: int = -1,
) -> dict:
    """
    Run greedy + beam inference over the validation set.

    Returns a dict::

        {
            "greedy": {"bleu1": float, "bleu4": float, "n_images": int},
            "beam":   {"bleu1": float, "bleu4": float, "n_images": int},
        }
    """
    refs = build_references(ann_file)
    transform = get_transform(
        False, cfg.img_size, cfg.crop_size, cfg.imagenet_mean, cfg.imagenet_std
    )

    with open(ann_file) as f:
        data = json.load(f)

    # Build {image_id → filename} mapping
    id2file: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
    image_ids = list(id2file.keys())

    if max_images > 0:
        image_ids = image_ids[:max_images]

    device = next(encoder.parameters()).device

    hyps_greedy: list[list[str]] = []
    hyps_beam:   list[list[str]] = []
    references:  list[list[list[str]]] = []

    for img_id in tqdm(image_ids, desc="Evaluating", dynamic_ncols=True, leave=True):
        if img_id not in refs:
            continue

        img_path = os.path.join(img_dir, id2file[img_id])
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)
        feat   = encoder(tensor)                           # (1, embed_size)

        # Greedy
        g_ids  = decoder.greedy_caption(feat, vocab, cfg.max_seq_len)
        g_toks = [vocab.idx2word.get(i, "<unk>") for i in g_ids]

        # Beam
        b_ids  = decoder.beam_search_caption(feat, vocab, beam_size, cfg.max_seq_len)
        b_toks = [vocab.idx2word.get(i, "<unk>") for i in b_ids]

        hyps_greedy.append(g_toks)
        hyps_beam.append(b_toks)
        references.append(refs[img_id])

    smooth = SmoothingFunction().method1

    def corpus_bleu_n(hyps: list[list[str]], n: int) -> float:
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        return corpus_bleu(references, hyps, weights=weights,
                           smoothing_function=smooth)

    return {
        "greedy": {
            "bleu1":    corpus_bleu_n(hyps_greedy, 1),
            "bleu4":    corpus_bleu_n(hyps_greedy, 4),
            "n_images": len(hyps_greedy),
        },
        "beam": {
            "bleu1":    corpus_bleu_n(hyps_beam, 1),
            "bleu4":    corpus_bleu_n(hyps_beam, 4),
            "n_images": len(hyps_beam),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_results(label: str, r: dict) -> None:
    bar = "─" * 58
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    for method in ("greedy", "beam"):
        m = r[method]
        print(f"  [{method:>6}]  BLEU-1: {m['bleu1']:.4f}   "
              f"BLEU-4: {m['bleu4']:.4f}   (n = {m['n_images']:,})")
    print(f"{bar}\n")


def _print_delta(r_main: dict, r_base: dict) -> None:
    print("  Δ (main − base):")
    for method in ("greedy", "beam"):
        d1 = r_main[method]["bleu1"] - r_base[method]["bleu1"]
        d4 = r_main[method]["bleu4"] - r_base[method]["bleu4"]
        print(f"  [{method:>6}]  ΔBLEU-1: {d1:+.4f}   ΔBLEU-4: {d4:+.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="BLEU evaluation for CNN-LSTM image captioning"
    )
    p.add_argument("--ckpt",       required=True,
                   help="Checkpoint to evaluate (.ckpt)")
    p.add_argument("--base_ckpt",  default=None,
                   help="Optional baseline checkpoint for Δ comparison")
    p.add_argument("--beam_size",  type=int, default=3)
    p.add_argument("--max_images", type=int, default=-1,
                   help="Cap number of val images (-1 = full set; "
                        "use e.g. 500 for a quick test)")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    device = cfg.device

    vocab = Vocabulary.load(cfg.vocab_path)
    print(f"[Eval] Vocabulary: {len(vocab):,} words | device: {device}")

    # ── Main checkpoint ───────────────────────────────────────────────────
    enc, dec, ckpt_cfg = load_model(args.ckpt, vocab, device)
    results = evaluate(
        enc, dec, vocab,
        ann_file   = cfg.val_ann,
        img_dir    = cfg.val_img_dir,
        cfg        = ckpt_cfg,
        beam_size  = args.beam_size,
        max_images = args.max_images,
    )
    _print_results(f"Checkpoint: {args.ckpt}", results)

    # ── Optional baseline comparison ──────────────────────────────────────
    if args.base_ckpt:
        b_enc, b_dec, b_cfg = load_model(args.base_ckpt, vocab, device)
        base_results = evaluate(
            b_enc, b_dec, vocab,
            ann_file   = cfg.val_ann,
            img_dir    = cfg.val_img_dir,
            cfg        = b_cfg,
            beam_size  = args.beam_size,
            max_images = args.max_images,
        )
        _print_results(f"Baseline: {args.base_ckpt}", base_results)
        _print_delta(results, base_results)


if __name__ == "__main__":
    main()