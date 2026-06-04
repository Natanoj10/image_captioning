#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# download_coco.sh — Download MS-COCO 2014 (images + annotations)
#
# Usage:
#   chmod +x download_coco.sh
#   ./download_coco.sh              # downloads to ./data/coco  (default)
#   ./download_coco.sh /path/dir    # custom destination
#
# After completion the directory will contain:
#   <dest>/
#   ├── annotations/
#   │   ├── captions_train2014.json
#   │   └── captions_val2014.json
#   ├── train2014/   (82 783 images, ~13 GB)
#   └── val2014/     (40 504 images,  ~6 GB)
#
# Requires: wget, unzip  (install with: sudo pacman -S wget unzip)
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

DEST="${1:-./data/coco}"
ANN_DIR="${DEST}/annotations"

# ── Colours ──────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}[✓]${RESET} $*"; }
info() { echo -e "${YELLOW}[→]${RESET} $*"; }

# ── Preflight ────────────────────────────────────────────────────
for cmd in wget unzip; do
    command -v "$cmd" &>/dev/null || {
        echo "ERROR: '$cmd' not found. Install with: sudo pacman -S $cmd"
        exit 1
    }
done

mkdir -p "$ANN_DIR"

echo "═══════════════════════════════════════════════════"
echo "  MS-COCO 2014 Downloader"
echo "  Destination : $DEST"
echo "═══════════════════════════════════════════════════"

# ── 1 / 3  Annotations (~241 MB) ────────────────────────────────
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
ANN_ZIP="${DEST}/annotations_trainval2014.zip"

if [ -f "${ANN_DIR}/captions_train2014.json" ] && \
   [ -f "${ANN_DIR}/captions_val2014.json" ]; then
    ok "[1/3] Annotations already present — skipping."
else
    info "[1/3] Downloading annotations (~241 MB) …"
    wget --continue --show-progress -q "${ANN_URL}" -O "${ANN_ZIP}"
    info "[1/3] Extracting …"
    unzip -q "${ANN_ZIP}" -d "${DEST}"
    rm "${ANN_ZIP}"
    ok "[1/3] Annotations ready."
fi

# ── 2 / 3  Train images (~13 GB) ────────────────────────────────
TRAIN_URL="http://images.cocodataset.org/zips/train2014.zip"
TRAIN_ZIP="${DEST}/train2014.zip"

if [ -d "${DEST}/train2014" ]; then
    N=$(ls -1q "${DEST}/train2014" | wc -l)
    ok "[2/3] train2014 already present (${N} files) — skipping."
else
    info "[2/3] Downloading train2014 (~13 GB) — grab a coffee ☕ …"
    wget --continue --show-progress -q "${TRAIN_URL}" -O "${TRAIN_ZIP}"
    info "[2/3] Extracting train2014 …"
    unzip -q "${TRAIN_ZIP}" -d "${DEST}"
    rm "${TRAIN_ZIP}"
    ok "[2/3] train2014 ready."
fi

# ── 3 / 3  Val images (~6 GB) ───────────────────────────────────
VAL_URL="http://images.cocodataset.org/zips/val2014.zip"
VAL_ZIP="${DEST}/val2014.zip"

if [ -d "${DEST}/val2014" ]; then
    N=$(ls -1q "${DEST}/val2014" | wc -l)
    ok "[3/3] val2014 already present (${N} files) — skipping."
else
    info "[3/3] Downloading val2014 (~6 GB) …"
    wget --continue --show-progress -q "${VAL_URL}" -O "${VAL_ZIP}"
    info "[3/3] Extracting val2014 …"
    unzip -q "${VAL_ZIP}" -d "${DEST}"
    rm "${VAL_ZIP}"
    ok "[3/3] val2014 ready."
fi

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  All done! Final layout:"
echo "  ${DEST}/"
echo "  ├── annotations/"
echo "  │   ├── captions_train2014.json"
echo "  │   └── captions_val2014.json"
echo "  ├── train2014/     (82 783 images)"
echo "  └── val2014/       (40 504 images)"
echo ""
echo "  Next steps:"
echo "    pip install --pre torch torchvision torchaudio \\"
echo "        --index-url https://download.pytorch.org/whl/nightly/cu128"
echo "    pip install -r requirements.txt"
echo "    python train.py"
echo "═══════════════════════════════════════════════════"
