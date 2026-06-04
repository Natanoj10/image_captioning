"""
api.py — FastAPI web interface for real-time image captioning.

The checkpoint to load is controlled by two environment variables
(both have sensible defaults):

    CAPTION_CKPT  — path to the .ckpt file  (default: checkpoints/best_model.ckpt)
    CAPTION_VOCAB — path to the vocab.pkl   (default: data/vocab.pkl)

Usage
-----
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Then open  http://localhost:8000  in any browser.

Endpoints
---------
    GET  /          → HTML single-page frontend
    POST /caption   → JSON {"greedy": str, "beam": str}
                      Body: multipart/form-data with field "file" (image)
                      Query: beam_size (int, default 3)
    GET  /health    → {"status": "ok", "model_loaded": bool, "vocab_size": int}
"""

import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from config import Config
from dataset import get_transform
from models import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary


# ─────────────────────────────────────────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────────────────────────────────────────

_state: dict = {
    "encoder":   None,
    "decoder":   None,
    "vocab":     None,
    "transform": None,
    "cfg":       None,
}


def _as_cfg(raw) -> Config:
    """Accept either a Config dataclass or the legacy cfg.__dict__ (plain dict)."""
    if isinstance(raw, Config):
        return raw
    valid_keys = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return Config(**{k: v for k, v in raw.items() if k in valid_keys})


def _load_models():
    """Load vocabulary and model weights into global _state."""
    base_cfg   = Config()
    ckpt_path  = os.environ.get(
        "CAPTION_CKPT",
        os.path.join(base_cfg.checkpoint_dir, "best_model.ckpt"),
    )
    vocab_path = os.environ.get("CAPTION_VOCAB", base_cfg.vocab_path)

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Vocabulary not found at '{vocab_path}'. "
            "Run train.py first (it builds the vocab automatically)."
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'. "
            "Run train.py first to produce a checkpoint."
        )

    vocab = Vocabulary.load(vocab_path)
    ckpt  = torch.load(ckpt_path, map_location=base_cfg.device, weights_only=False)
    mcfg  = _as_cfg(ckpt["cfg"])

    enc = EncoderCNN(mcfg.embed_size, mcfg.encoder_backbone).to(base_cfg.device)
    dec = DecoderRNN(
        mcfg.embed_size, mcfg.hidden_size, len(vocab),
        mcfg.num_layers, mcfg.dropout, mcfg.cell_type,
    ).to(base_cfg.device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval()
    dec.eval()

    _state["encoder"]   = enc
    _state["decoder"]   = dec
    _state["vocab"]     = vocab
    _state["cfg"]       = mcfg
    _state["transform"] = get_transform(
        False,
        mcfg.img_size, mcfg.crop_size,
        mcfg.imagenet_mean, mcfg.imagenet_std,
    )

    print(f"[API] Loaded: {ckpt_path}")
    print(f"[API] Backbone={mcfg.encoder_backbone}  "
          f"cell={mcfg.cell_type}  vocab={len(vocab):,}")


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(
    title="Image Captioning API",
    version="1.0",
    description="CNN-LSTM encoder-decoder · MS-COCO · UG Deep Learning",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def _predict(img: Image.Image, beam_size: int) -> dict[str, str]:
    cfg     = Config()
    mcfg    = _state["cfg"]
    tensor  = _state["transform"](img).unsqueeze(0).to(cfg.device)

    with torch.no_grad():
        feat  = _state["encoder"](tensor)
        g_ids = _state["decoder"].greedy_caption(
            feat, _state["vocab"], mcfg.max_seq_len
        )
        b_ids = _state["decoder"].beam_search_caption(
            feat, _state["vocab"], beam_size, mcfg.max_seq_len
        )

    return {
        "greedy": _state["vocab"].decode(g_ids),
        "beam":   _state["vocab"].decode(b_ids),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return _HTML_PAGE


@app.post(
    "/caption",
    summary="Generate a caption for an uploaded image",
    response_description='{"greedy": "...", "beam": "..."}',
)
async def caption(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, …)"),
    beam_size: int   = Query(3, ge=1, le=10,
                             description="Beam size for beam-search decoding"),
):
    if _state["encoder"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 12 MB)")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image file")

    result = _predict(img, beam_size)
    return JSONResponse(content=result)


@app.get("/health", summary="Liveness probe")
async def health():
    vocab = _state["vocab"]
    return {
        "status":       "ok",
        "model_loaded": _state["encoder"] is not None,
        "vocab_size":   len(vocab) if vocab else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Embedded HTML frontend  (single-file, zero external JS dependencies)
# ─────────────────────────────────────────────────────────────────────────────

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Neural Captioning</title>
<style>
:root {
  --bg:      #0d1117;
  --surface: #161b22;
  --border:  #30363d;
  --accent:  #58a6ff;
  --accent2: #bc8cff;
  --text:    #c9d1d9;
  --sub:     #8b949e;
  --ok:      #3fb950;
  --err:     #f85149;
  --r: 12px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem 1rem 4rem;
}

h1 {
  font-size: 2.1rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: .3rem;
}
.subtitle { color: var(--sub); font-size: .88rem; margin-bottom: 2rem; }

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 2rem;
  width: 100%;
  max-width: 640px;
}

/* ── Drop zone ── */
.drop-zone {
  border: 2px dashed var(--border);
  border-radius: var(--r);
  padding: 2.5rem 1rem;
  text-align: center;
  cursor: pointer;
  position: relative;
  transition: border-color .2s, background .2s;
}
.drop-zone:hover, .drop-zone.drag-over {
  border-color: var(--accent);
  background: rgba(88,166,255,.06);
}
.drop-zone input[type=file] {
  position: absolute; inset: 0; opacity: 0; cursor: pointer;
}
.drop-icon { font-size: 2.6rem; margin-bottom: .5rem; }
.drop-label { color: var(--sub); font-size: .88rem; }
.drop-label span { color: var(--accent); font-weight: 600; }

/* ── Preview ── */
#preview-wrap {
  display: none;
  margin-top: 1.25rem;
  border-radius: var(--r);
  overflow: hidden;
  text-align: center;
  max-height: 340px;
}
#preview-img {
  max-width: 100%; max-height: 340px;
  object-fit: contain;
  border-radius: var(--r);
}

/* ── Beam slider ── */
.beam-row {
  display: flex; align-items: center; gap: .75rem;
  margin-top: 1.25rem;
}
.beam-row label { font-size: .84rem; color: var(--sub); white-space: nowrap; }
input[type=range] { flex: 1; accent-color: var(--accent); }
#beam-val { font-size: .84rem; font-weight: 700; color: var(--accent); min-width: 1.2rem; }

/* ── Button ── */
#submit-btn {
  margin-top: 1.5rem;
  width: 100%;
  padding: .85rem;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #fff;
  border: none;
  border-radius: var(--r);
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: opacity .2s, transform .1s;
}
#submit-btn:hover:not(:disabled) { opacity: .88; }
#submit-btn:active:not(:disabled) { transform: scale(.99); }
#submit-btn:disabled { opacity: .4; cursor: not-allowed; }

/* ── Spinner ── */
.spinner {
  display: none; margin: 1.1rem auto;
  width: 36px; height: 36px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Error ── */
.error-box {
  display: none;
  margin-top: 1rem;
  padding: .8rem 1rem;
  background: rgba(248,81,73,.1);
  border: 1px solid rgba(248,81,73,.35);
  border-radius: 8px;
  color: var(--err);
  font-size: .88rem;
}

/* ── Results ── */
#result-section { display: none; margin-top: 1.5rem; }

.result-label {
  font-size: .73rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--sub);
  margin-bottom: .35rem;
}
.caption-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: .9rem 3.5rem .9rem 1rem;
  font-size: 1rem;
  line-height: 1.6;
  color: var(--text);
  margin-bottom: 1rem;
  position: relative;
  min-height: 3.2rem;
}
.method-tag {
  position: absolute;
  top: .45rem; right: .6rem;
  font-size: .62rem; font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .07em;
  padding: .15rem .45rem;
  border-radius: 4px;
}
.tag-greedy { background: rgba(63,185,80,.14);  color: var(--ok); }
.tag-beam   { background: rgba(88,166,255,.14); color: var(--accent); }

/* ── Copy buttons ── */
.copy-btn {
  position: absolute;
  bottom: .45rem; right: .55rem;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--sub);
  border-radius: 6px;
  padding: .15rem .45rem;
  font-size: .72rem;
  cursor: pointer;
  transition: color .15s, border-color .15s;
}
.copy-btn:hover { color: var(--accent); border-color: var(--accent); }

footer {
  margin-top: 3rem;
  color: var(--sub);
  font-size: .76rem;
  text-align: center;
}
</style>
</head>
<body>

<h1>Neural Captioning</h1>
<p class="subtitle">
  CNN‑LSTM Encoder‑Decoder &nbsp;·&nbsp; MS-COCO &nbsp;·&nbsp;
  Universidad de Guanajuato
</p>

<div class="card">

  <!-- Drop zone -->
  <div class="drop-zone" id="drop-zone">
    <input type="file" id="file-input" accept="image/*" />
    <div class="drop-icon">🖼️</div>
    <p class="drop-label">
      <span>Click to upload</span> or drag &amp; drop an image here<br />
      <span style="font-size:.8rem;color:var(--sub)">JPEG · PNG · WebP · BMP</span>
    </p>
  </div>

  <!-- Preview -->
  <div id="preview-wrap">
    <img id="preview-img" src="" alt="preview" />
  </div>

  <!-- Beam slider -->
  <div class="beam-row">
    <label for="beam-slider">Beam size</label>
    <input type="range" id="beam-slider" min="1" max="7" value="3" />
    <span id="beam-val">3</span>
  </div>

  <button id="submit-btn" disabled>Generate Description</button>

  <div class="spinner" id="spinner"></div>
  <div class="error-box" id="error-box"></div>

  <!-- Results -->
  <div id="result-section">
    <div class="result-label">Greedy Decoding</div>
    <div class="caption-box" style="padding-top:1.5rem">
      <span class="method-tag tag-greedy">Greedy</span>
      <span id="cap-greedy">—</span>
      <button class="copy-btn" onclick="copyText('cap-greedy')">copy</button>
    </div>

    <div class="result-label">
      Beam Search &nbsp;(k = <span id="beam-k-lbl">3</span>)
    </div>
    <div class="caption-box" style="padding-top:1.5rem">
      <span class="method-tag tag-beam">Beam</span>
      <span id="cap-beam">—</span>
      <button class="copy-btn" onclick="copyText('cap-beam')">copy</button>
    </div>
  </div>

</div><!-- /card -->

<footer>
  Aprendizaje Profundo &nbsp;·&nbsp; Ingeniería de Datos e IA &nbsp;·&nbsp; UG &nbsp;·&nbsp; 2025
</footer>

<script>
// ── Elements ──────────────────────────────────────────────────────
const fileInput  = document.getElementById('file-input');
const dropZone   = document.getElementById('drop-zone');
const previewImg = document.getElementById('preview-img');
const previewWrap= document.getElementById('preview-wrap');
const submitBtn  = document.getElementById('submit-btn');
const spinner    = document.getElementById('spinner');
const errorBox   = document.getElementById('error-box');
const resultSec  = document.getElementById('result-section');
const beamSlider = document.getElementById('beam-slider');
const beamVal    = document.getElementById('beam-val');
const beamKLbl   = document.getElementById('beam-k-lbl');
const capGreedy  = document.getElementById('cap-greedy');
const capBeam    = document.getElementById('cap-beam');

let selectedFile = null;

// ── Beam slider ───────────────────────────────────────────────────
beamSlider.addEventListener('input', () => {
  beamVal.textContent  = beamSlider.value;
  beamKLbl.textContent = beamSlider.value;
});

// ── Drag-and-drop ─────────────────────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(f) {
  selectedFile = f;
  const url = URL.createObjectURL(f);
  previewImg.src = url;
  previewWrap.style.display = 'block';
  submitBtn.disabled         = false;
  resultSec.style.display    = 'none';
  errorBox.style.display     = 'none';
}

// ── Submit ────────────────────────────────────────────────────────
submitBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  submitBtn.disabled      = true;
  spinner.style.display   = 'block';
  errorBox.style.display  = 'none';
  resultSec.style.display = 'none';

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const res = await fetch(`/caption?beam_size=${beamSlider.value}`, {
      method: 'POST',
      body: fd,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    capGreedy.textContent   = data.greedy || '(empty)';
    capBeam.textContent     = data.beam   || '(empty)';
    resultSec.style.display = 'block';
  } catch (e) {
    errorBox.textContent   = 'Error: ' + e.message;
    errorBox.style.display = 'block';
  } finally {
    spinner.style.display = 'none';
    submitBtn.disabled    = false;
  }
});

// ── Copy helper ───────────────────────────────────────────────────
function copyText(id) {
  const el = document.getElementById(id);
  navigator.clipboard.writeText(el.textContent)
    .then(() => {
      const btn = el.parentElement.querySelector('.copy-btn');
      const orig = btn.textContent;
      btn.textContent = '✓';
      setTimeout(() => { btn.textContent = orig; }, 1200);
    });
}
</script>
</body>
</html>
"""