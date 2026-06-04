"""
validate_external.py — Qualitative validation with ≥10 external images.

Loads two model checkpoints (base + best variant) and generates captions
for every image in `imagenes_validation/`, then produces:

  results/external_validation.png
      Dark-themed matrix figure: one row per image
      columns: thumbnail | base greedy | base beam | variant greedy | variant beam

  results/external_validation.csv
      Same data as plain text — easy to paste into the report.

Usage
-----
    python validate_external.py \
        --base    checkpoints/base_model.ckpt \
        --variant checkpoints/best_model.ckpt

If both checkpoints point to the same file (single-model run), the
two caption columns will be identical — that's fine for a first test.
"""

import argparse
import csv
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import torch

from config import Config
from dataset import get_transform
from models import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Dark-theme palette
BG_COLOR     = "#0d1117"
SURFACE_COLOR= "#161b22"
HEADER_COLOR = "#e94560"
BASE_BG      = "#0f3460"
VAR_BG       = "#533483"
TEXT_COLOR   = "#c9d1d9"
BORDER_COLOR = "#30363d"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _as_cfg(raw) -> Config:
    """Accept either a Config dataclass or the legacy cfg.__dict__ (plain dict)."""
    if isinstance(raw, Config):
        return raw
    valid_keys = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return Config(**{k: v for k, v in raw.items() if k in valid_keys})


def _load_model(ckpt_path: str, vocab: Vocabulary, device: str):
    """Restore encoder + decoder. Returns (encoder, decoder, cfg)."""
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


@torch.no_grad()
def _get_captions(
    encoder, decoder, vocab: Vocabulary,
    transform, img_path: str,
    beam_size: int, max_len: int, device: str,
) -> tuple[str, str]:
    """Return (greedy_caption, beam_caption) for a single image file."""
    img    = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    feat   = encoder(tensor)

    g_ids = decoder.greedy_caption(feat, vocab, max_len)
    b_ids = decoder.beam_search_caption(feat, vocab, beam_size, max_len)

    return vocab.decode(g_ids), vocab.decode(b_ids)


def _wrap(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(text if text else "—", width))


# ─────────────────────────────────────────────────────────────────────────────
# Figure builder
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_figure(rows: list[dict], save_path: str) -> None:
    """
    rows: list of dicts with keys:
        path, base_greedy, base_beam, var_greedy, var_beam
    Saves a dark-themed matrix PNG to save_path.
    """
    n   = len(rows)
    # 1 header row + n data rows; each ~2.8 inches tall
    fig = plt.figure(
        figsize=(20, 1.4 + n * 2.8),
        facecolor=BG_COLOR,
    )

    # 5 columns: thumbnail | base-greedy | base-beam | var-greedy | var-beam
    n_rows_gs = n + 1   # +1 for header
    gs = GridSpec(
        n_rows_gs, 5, figure=fig,
        height_ratios=[0.45] + [1.0] * n,
        hspace=0.06, wspace=0.04,
        left=0.02, right=0.98,
        top=0.97,  bottom=0.03,
    )

    # ── Header row ───────────────────────────────────────────────────────
    header_labels = [
        "Image",
        "Base — Greedy",
        "Base — Beam",
        "Variant — Greedy",
        "Variant — Beam",
    ]
    header_bgs = [SURFACE_COLOR, BASE_BG, BASE_BG, VAR_BG, VAR_BG]

    for ci, (lbl, bg) in enumerate(zip(header_labels, header_bgs)):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_facecolor(bg)
        ax.text(
            0.5, 0.5, lbl,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white",
            transform=ax.transAxes,
        )
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(HEADER_COLOR); sp.set_linewidth(1.2)

    # ── Data rows ────────────────────────────────────────────────────────
    for ri, row in enumerate(rows):
        row_idx = ri + 1   # offset for header

        # --- Thumbnail ---
        ax_img = fig.add_subplot(gs[row_idx, 0])
        try:
            img_pil = Image.open(row["path"]).convert("RGB")
            img_pil.thumbnail((220, 220))
            ax_img.imshow(img_pil)
        except Exception:
            ax_img.text(0.5, 0.5, "[error]", ha="center", va="center",
                        color="red", transform=ax_img.transAxes)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        fname = Path(row["path"]).stem
        fname_disp = (fname[:22] + "…") if len(fname) > 23 else fname
        ax_img.set_title(fname_disp, fontsize=7.5, color=TEXT_COLOR, pad=3)
        for sp in ax_img.spines.values():
            sp.set_edgecolor(BORDER_COLOR)

        # --- Caption cells ---
        captions = [
            row["base_greedy"], row["base_beam"],
            row["var_greedy"],  row["var_beam"],
        ]
        cell_bgs = [BASE_BG, BASE_BG, VAR_BG, VAR_BG]

        for ci, (txt, bg) in enumerate(zip(captions, cell_bgs), start=1):
            ax = fig.add_subplot(gs[row_idx, ci])
            ax.set_facecolor(bg)
            ax.text(
                0.5, 0.5, _wrap(txt, width=32),
                ha="center", va="center",
                fontsize=8.5, color=TEXT_COLOR,
                transform=ax.transAxes,
                linespacing=1.4,
            )
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER_COLOR)

    # ── Legend ───────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=BASE_BG,  label="Base model (CNN-LSTM, frozen)"),
        mpatches.Patch(color=VAR_BG,   label="Best variant (fine-tuned / GRU / Beam)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=2, fontsize=8.5,
        framealpha=0.25, facecolor=SURFACE_COLOR,
        labelcolor=TEXT_COLOR,
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.suptitle(
        "External Validation — Base vs. Variant Captions",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.999,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Validate] Figure  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], save_path: str) -> None:
    fieldnames = ["filename", "base_greedy", "base_beam", "var_greedy", "var_beam"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({
                "filename":    Path(row["path"]).name,
                "base_greedy": row["base_greedy"],
                "base_beam":   row["base_beam"],
                "var_greedy":  row["var_greedy"],
                "var_beam":    row["var_beam"],
            })
    print(f"[Validate] CSV     → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Qualitative external validation for image captioning"
    )
    p.add_argument("--base",      required=True,
                   help="Base checkpoint (.ckpt) — frozen CNN-LSTM")
    p.add_argument("--variant",   required=True,
                   help="Best variant checkpoint (.ckpt) — fine-tuned / GRU / etc.")
    p.add_argument("--beam_size", type=int, default=3)
    p.add_argument("--max_len",   type=int, default=25)
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    device = cfg.device

    vocab = Vocabulary.load(cfg.vocab_path)
    print(f"[Validate] Vocab: {len(vocab):,} words | device: {device}")

    # ── Load both models ──────────────────────────────────────────────────
    print("[Validate] Loading base model …")
    b_enc, b_dec, b_cfg = _load_model(args.base, vocab, device)
    transform = get_transform(
        False, b_cfg.img_size, b_cfg.crop_size,
        b_cfg.imagenet_mean, b_cfg.imagenet_std,
    )

    print("[Validate] Loading variant model …")
    v_enc, v_dec, _     = _load_model(args.variant, vocab, device)

    # ── Gather external images ────────────────────────────────────────────
    val_dir   = Path(cfg.val_ext_dir)
    img_paths = sorted(
        p for p in val_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )

    if not img_paths:
        raise FileNotFoundError(
            f"No images found in '{val_dir}'.\n"
            f"Place ≥10 diverse images there before running this script.\n"
            f"Supported formats: {SUPPORTED_EXTS}"
        )
    if len(img_paths) < 10:
        print(
            f"[WARN] Only {len(img_paths)} images found in '{val_dir}'. "
            "The project requires ≥10 for full credit."
        )

    # ── Run inference ─────────────────────────────────────────────────────
    rows: list[dict] = []
    for p in img_paths:
        print(f"  → {p.name}")
        bg, bb = _get_captions(
            b_enc, b_dec, vocab, transform, str(p),
            args.beam_size, args.max_len, device,
        )
        vg, vb = _get_captions(
            v_enc, v_dec, vocab, transform, str(p),
            args.beam_size, args.max_len, device,
        )
        rows.append({
            "path":        str(p),
            "base_greedy": bg, "base_beam": bb,
            "var_greedy":  vg, "var_beam":  vb,
        })

    # ── Save outputs ──────────────────────────────────────────────────────
    os.makedirs(cfg.results_dir, exist_ok=True)
    fig_path = os.path.join(cfg.results_dir, "external_validation.png")
    csv_path = os.path.join(cfg.results_dir, "external_validation.csv")
    make_comparison_figure(rows, fig_path)
    write_csv(rows, csv_path)

    # ── Console summary ───────────────────────────────────────────────────
    bar = "─" * 80
    print(f"\n{bar}")
    print(f"  {'File':28s}  {'Base (beam)':30s}  Variant (beam)")
    print(bar)
    for row in rows:
        name = Path(row["path"]).name[:27]
        bc   = row["base_beam"][:29]
        vc   = row["var_beam"][:35]
        print(f"  {name:28s}  {bc:30s}  {vc}")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()