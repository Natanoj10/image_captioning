"""
train.py — Full training pipeline for the CNN-LSTM image captioning model.

Usage
-----
# Build vocab + train from scratch (10 epochs, freeze CNN for first 5):
    python train.py

# Quick smoke-test with 2 000 samples:
    python train.py --max_samples 2000 --epochs 2

# Resume from a checkpoint:
    python train.py --resume checkpoints/epoch_05.ckpt

# Train GRU variant:
    python train.py --cell_type gru --tag gru_variant
"""

import argparse
import math
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import CocoDataset, get_transform
from models import DecoderRNN, EncoderCNN
from vocabulary import Vocabulary, build_vocab


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(cfg: Config, vocab: Vocabulary):
    collate = partial(CocoDataset.collate_fn, pad_idx=vocab.pad_idx)

    train_ds = CocoDataset(
        cfg.train_img_dir, cfg.train_ann, vocab,
        transform=get_transform(True,  cfg.img_size, cfg.crop_size,
                                cfg.imagenet_mean, cfg.imagenet_std),
        max_samples=cfg.max_train_samples,
    )
    val_ds = CocoDataset(
        cfg.val_img_dir, cfg.val_ann, vocab,
        transform=get_transform(False, cfg.img_size, cfg.crop_size,
                                cfg.imagenet_mean, cfg.imagenet_std),
    )

    kw = dict(collate_fn=collate, num_workers=cfg.num_workers,
              pin_memory=cfg.pin_memory)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                              drop_last=True, **kw)   # drop_last evita batch=1 → falla BN1d
    val_loader   = DataLoader(val_ds,   cfg.batch_size, shuffle=False, **kw)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(encoder, decoder, loader, criterion,
                enc_opt, dec_opt, scaler, device, cfg, epoch):
    encoder.train()
    decoder.train()

    total_loss, total_tok = 0.0, 0
    pbar = tqdm(loader, desc=f"Train Ep{epoch:02d}", dynamic_ncols=True,
                leave=False)

    for step, (images, captions, lengths) in enumerate(pbar, 1):
        images   = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        lengths  = lengths.to(device, non_blocking=True)

        # dec_len = lengths - 1  (shared by encoder input packing and target packing)
        dec_len = (lengths - 1).clamp(min=1)

        enc_opt.zero_grad(set_to_none=True)
        dec_opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            features = encoder(images)                          # (B, E)
            logits   = decoder(features, captions, dec_len)    # (Σt, V)

            # Pack targets: caption[:, 1:] = [w1, …, wN, <end>]
            targets = captions[:, 1:].contiguous()
            tgt_packed = pack_padded_sequence(
                targets, dec_len.cpu(), batch_first=True, enforce_sorted=True
            )
            loss = criterion(logits, tgt_packed.data)

        scaler.scale(loss).backward()

        # Gradient clipping (important for RNNs)
        scaler.unscale_(dec_opt)
        nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        if epoch >= cfg.finetune_epoch:
            scaler.unscale_(enc_opt)
            nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)

        scaler.step(dec_opt)
        if epoch >= cfg.finetune_epoch:
            scaler.step(enc_opt)
        scaler.update()

        n_tok        = tgt_packed.data.size(0)
        total_loss  += loss.item() * n_tok
        total_tok   += n_tok

        if step % cfg.log_step == 0:
            avg = total_loss / total_tok
            pbar.set_postfix(loss=f"{avg:.4f}", pp=f"{math.exp(avg):.2f}")

    return total_loss / total_tok


@torch.no_grad()
def val_epoch(encoder, decoder, loader, criterion, device, cfg):
    encoder.eval()
    decoder.eval()

    total_loss, total_tok = 0.0, 0

    for images, captions, lengths in tqdm(loader, desc="  Val   ",
                                          dynamic_ncols=True, leave=False):
        images   = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        lengths  = lengths.to(device, non_blocking=True)

        dec_len = (lengths - 1).clamp(min=1)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            features = encoder(images)
            logits   = decoder(features, captions, dec_len)

            targets = captions[:, 1:].contiguous()
            tgt_packed = pack_padded_sequence(
                targets, dec_len.cpu(), batch_first=True, enforce_sorted=True
            )
            loss = criterion(logits, tgt_packed.data)

        n_tok        = tgt_packed.data.size(0)
        total_loss  += loss.item() * n_tok
        total_tok   += n_tok

    return total_loss / total_tok


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(encoder, decoder, enc_opt, dec_opt,
              epoch, val_loss, cfg, filename):
    path = os.path.join(cfg.checkpoint_dir, filename)
    torch.save({
        "epoch":    epoch,
        "encoder":  encoder.state_dict(),
        "decoder":  decoder.state_dict(),
        "enc_opt":  enc_opt.state_dict(),
        "dec_opt":  dec_opt.state_dict(),
        "val_loss": val_loss,
        "cfg":      cfg,
    }, path)
    return path


def plot_curves(train_losses, val_losses, results_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, "b-o", label="Train")
    ax1.plot(epochs, val_losses,   "r-o", label="Val")
    ax1.set(xlabel="Epoch", ylabel="Cross-Entropy Loss",
            title="Loss (Train vs Val)")
    ax1.legend(); ax1.grid(alpha=0.3)

    train_pp = [math.exp(l) for l in train_losses]
    val_pp   = [math.exp(l) for l in val_losses]
    ax2.plot(epochs, train_pp, "b-o", label="Train")
    ax2.plot(epochs, val_pp,   "r-o", label="Val")
    ax2.set(xlabel="Epoch", ylabel="Perplexity  (PP = e^Loss)",
            title="Perplexity (Train vs Val)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train image captioning model")
    p.add_argument("--resume",      default=None,  help="Checkpoint to resume from")
    p.add_argument("--epochs",      type=int,      default=None, help="Override num_epochs")
    p.add_argument("--batch",       type=int,      default=None, help="Override batch_size")
    p.add_argument("--max_samples", type=int,      default=None, help="Limit training set size")
    p.add_argument("--cell_type",   default=None,  choices=["lstm", "gru"])
    p.add_argument("--tag",         default="",    help="Optional label for checkpoint names")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    # CLI overrides
    if args.epochs:      cfg.num_epochs        = args.epochs
    if args.batch:       cfg.batch_size        = args.batch
    if args.max_samples: cfg.max_train_samples = args.max_samples
    if args.cell_type:   cfg.cell_type         = args.cell_type

    device = torch.device(cfg.device)
    torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print(" Image Captioning — Training")
    print(f"  Device   : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  AMP      : {cfg.use_amp}")
    print(f"  Backbone : {cfg.encoder_backbone}  Cell: {cfg.cell_type.upper()}")
    print(f"  Epochs   : {cfg.num_epochs}  BatchSize: {cfg.batch_size}")
    print(f"  Fine-tune encoder from epoch {cfg.finetune_epoch}")
    print("=" * 60)

    # ── Vocabulary ────────────────────────────────────────────────────────
    if os.path.exists(cfg.vocab_path):
        vocab = Vocabulary.load(cfg.vocab_path)
        print(f"[Vocab] Loaded {len(vocab):,} words from {cfg.vocab_path}")
    else:
        print("[Vocab] Building vocabulary …")
        vocab = build_vocab(cfg.train_ann, cfg.vocab_threshold, cfg.vocab_path)

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_loader, val_loader = get_loaders(cfg, vocab)
    print(f"[Data]  Train {len(train_loader.dataset):,} samples | "
          f"Val {len(val_loader.dataset):,} samples")

    # ── Models ────────────────────────────────────────────────────────────
    encoder = EncoderCNN(cfg.embed_size, cfg.encoder_backbone).to(device)
    decoder = DecoderRNN(cfg.embed_size, cfg.hidden_size, len(vocab),
                         cfg.num_layers, cfg.dropout, cfg.cell_type).to(device)

    enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    dec_params = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"[Model] Encoder {enc_params:.2f}M params | Decoder {dec_params:.2f}M params")

    # ── Optimisers ────────────────────────────────────────────────────────
    # Encoder optimiser only trains the unfrozen params (adaptation head initially)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=cfg.learning_rate)
    enc_opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=cfg.finetune_lr,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    start_epoch   = 1
    train_losses  : list[float] = []
    val_losses    : list[float] = []
    best_val_loss = float("inf")
    tag = f"_{args.tag}" if args.tag else ""

    # ── Optional resume ───────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        enc_opt.load_state_dict(ckpt["enc_opt"])
        dec_opt.load_state_dict(ckpt["dec_opt"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"[Resume] epoch {ckpt['epoch']}  val_loss={best_val_loss:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.num_epochs + 1):

        # ── Fine-tuning: unfreeze backbone at finetune_epoch ──────────
        if epoch == cfg.finetune_epoch:
            encoder.unfreeze_backbone()
            # Recreate optimiser so the unfrozen backbone params are included
            enc_opt = torch.optim.Adam(
                encoder.parameters(), lr=cfg.finetune_lr
            )

        # ── Save "base" checkpoint just before fine-tuning starts ──────
        if epoch == cfg.finetune_epoch:
            path = save_ckpt(encoder, decoder, enc_opt, dec_opt,
                             epoch - 1, best_val_loss, cfg,
                             f"base_model{tag}.ckpt")
            print(f"[Base] Saved pre-finetune checkpoint → {path}")

        t0 = time.perf_counter()
        tr_loss = train_epoch(encoder, decoder, train_loader, criterion,
                              enc_opt, dec_opt, scaler, device, cfg, epoch)
        vl_loss = val_epoch(encoder, decoder, val_loader, criterion, device, cfg)
        elapsed = time.perf_counter() - t0

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(
            f"Epoch {epoch:02d}/{cfg.num_epochs} | "
            f"TrainLoss {tr_loss:.4f} PP {math.exp(tr_loss):.2f} | "
            f"ValLoss {vl_loss:.4f} PP {math.exp(vl_loss):.2f} | "
            f"{elapsed:.0f}s"
        )

        # ── Save best model ────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            p = save_ckpt(encoder, decoder, enc_opt, dec_opt,
                          epoch, vl_loss, cfg, f"best_model{tag}.ckpt")
            print(f"  ✓  best model → {p}")

        # ── Save per-epoch checkpoint ──────────────────────────────────
        save_ckpt(encoder, decoder, enc_opt, dec_opt,
                  epoch, vl_loss, cfg, f"epoch_{epoch:02d}{tag}.ckpt")

        # ── Update plots & CSV after every epoch ──────────────────────
        plot_curves(train_losses, val_losses, cfg.results_dir)

        csv_path = os.path.join(cfg.results_dir, f"metrics{tag}.csv")
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,train_pp,val_loss,val_pp\n")
            for i, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
                f.write(f"{i},{tl:.6f},{math.exp(tl):.4f},"
                        f"{vl:.6f},{math.exp(vl):.4f}\n")

    print(f"\nTraining done. Best Val Loss: {best_val_loss:.4f} "
          f"(PP={math.exp(best_val_loss):.2f})")


if __name__ == "__main__":
    main()