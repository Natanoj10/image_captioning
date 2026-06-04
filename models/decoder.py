"""
models/decoder.py — RNN Decoder for image captioning.

Key design choices
------------------
1. Hidden state initialised from the image feature via learned projections
   (init_h / init_c), so no special "visual token" is prepended.

2. Teacher Forcing + pack_padded_sequence
   Input  at step t : ground-truth token at t-1  → captions[:, :-1]
   Target at step t : ground-truth token at t    → captions[:,  1:]
   Both packed with dec_len = lengths - 1 (caller pre-computes this).

3. Supports LSTM (default) and GRU via `cell_type` — makes it trivial to
   switch as a variant.

4. Greedy and Beam-Search inference for captioning unseen images.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DecoderRNN(nn.Module):

    def __init__(
        self,
        embed_size:  int,
        hidden_size: int,
        vocab_size:  int,
        num_layers:  int   = 1,
        dropout:     float = 0.5,
        cell_type:   str   = "lstm",
    ):
        super().__init__()
        self.cell_type   = cell_type.lower()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size

        assert self.cell_type in ("lstm", "gru"), \
            "cell_type must be 'lstm' or 'gru'"

        # ── Embedding ─────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # ── Hidden-state initialisation (from visual features) ────────
        self.init_h = nn.Linear(embed_size, hidden_size)
        if self.cell_type == "lstm":
            self.init_c = nn.Linear(embed_size, hidden_size)

        # ── RNN cell ──────────────────────────────────────────────────
        rnn_cls = nn.LSTM if self.cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            embed_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Output head ───────────────────────────────────────────────
        self.fc      = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    # ── Internal helpers ─────────────────────────────────────────────────
    def _init_hidden(self, features: torch.Tensor):
        """
        features : (B, embed_size)
        Returns (h0, c0) for LSTM  |  h0 for GRU
        Each tensor: (num_layers, B, hidden_size)
        """
        h = torch.tanh(self.init_h(features))                      # (B, H)
        h = h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        if self.cell_type == "lstm":
            c = torch.tanh(self.init_c(features))
            c = c.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
            return (h, c)
        return h

    # ── Training forward (Teacher Forcing) ───────────────────────────────
    def forward(
        self,
        features: torch.Tensor,   # (B, embed_size) — from EncoderCNN
        captions: torch.Tensor,   # (B, T)          — padded, with <start>/<end>
        dec_len:  torch.Tensor,   # (B,)            — lengths - 1  (pre-computed)
    ) -> torch.Tensor:
        """
        Teacher Forcing + pack_padded_sequence.

        captions[:, :-1]  =  [<start>, w1, …, wN]          (length T-1) → input
        captions[:,  1:]  =  [w1, …, wN, <end>]            (length T-1) → target
        dec_len           =  lengths - 1  (passed in by caller)

        Returns
        -------
        logits : (Σ dec_len_i, vocab_size)   — packed, ready for CrossEntropyLoss
        """
        embeddings  = self.dropout(self.embedding(captions[:, :-1]))  # (B, T-1, E)
        dec_len_cpu = dec_len.cpu()

        hidden     = self._init_hidden(features)
        packed_in  = pack_padded_sequence(
            embeddings, dec_len_cpu, batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.rnn(packed_in, hidden)
        hiddens, _    = pad_packed_sequence(packed_out, batch_first=True)  # (B, T-1, H)

        logits = self.fc(self.dropout(hiddens))                    # (B, T-1, V)

        # Re-pack so we return only real (non-padding) positions
        packed_logits = pack_padded_sequence(
            logits, dec_len_cpu, batch_first=True, enforce_sorted=True
        )
        return packed_logits.data                                  # (Σ t_i, V)

    # ── Greedy inference ─────────────────────────────────────────────────
    @torch.no_grad()
    def greedy_caption(
        self,
        feature: torch.Tensor,   # (1, embed_size)
        vocab,
        max_len: int = 25,
    ) -> list[int]:
        """Greedy decoding; returns word index list (no special tokens)."""
        hidden = self._init_hidden(feature)
        word   = torch.tensor([[vocab.start_idx]], device=feature.device)
        result: list[int] = []

        for _ in range(max_len):
            emb = self.dropout(self.embedding(word))   # (1, 1, E)
            out, hidden = self.rnn(emb, hidden)        # (1, 1, H)
            logit = self.fc(out.squeeze(1))            # (1, V)
            idx   = logit.argmax(dim=-1).item()

            if idx == vocab.end_idx:
                break
            result.append(idx)
            word = torch.tensor([[idx]], device=feature.device)

        return result

    # ── Beam Search inference ────────────────────────────────────────────
    @torch.no_grad()
    def beam_search_caption(
        self,
        feature:   torch.Tensor,   # (1, embed_size)
        vocab,
        beam_size: int = 3,
        max_len:   int = 25,
    ) -> list[int]:
        """
        Beam Search decoding.

        Each beam is a dict:
            score  : cumulative negative log-probability (lower = better)
            tokens : list of generated token indices (may include <end>)
            hidden : RNN hidden state after the last token

        Returns the token list of the best completed hypothesis
        (special tokens stripped).
        """
        device = feature.device
        hidden = self._init_hidden(feature)

        # ── Seed with <start> ─────────────────────────────────────────
        word   = torch.tensor([[vocab.start_idx]], device=device)
        emb    = self.dropout(self.embedding(word))
        out, h = self.rnn(emb, hidden)
        lp     = torch.log_softmax(self.fc(out.squeeze(1)), dim=-1)[0]  # (V,)
        top_scores, top_tokens = lp.topk(beam_size)

        beams: list[dict] = [
            {"score": -s.item(), "tokens": [t.item()], "hidden": h}
            for s, t in zip(top_scores, top_tokens)
        ]
        completed: list[tuple[float, list[int]]] = []

        # ── Expand beams ──────────────────────────────────────────────
        for _ in range(max_len - 1):
            candidates: list[dict] = []

            for beam in beams:
                last = beam["tokens"][-1]

                if last == vocab.end_idx:
                    completed.append((beam["score"], beam["tokens"][:-1]))
                    continue

                word   = torch.tensor([[last]], device=device)
                emb    = self.dropout(self.embedding(word))
                out, h_new = self.rnn(emb, beam["hidden"])
                lp_b   = torch.log_softmax(self.fc(out.squeeze(1)), dim=-1)[0]
                top_lp, top_tok = lp_b.topk(beam_size)

                for lp_i, tok_i in zip(top_lp.tolist(), top_tok.tolist()):
                    candidates.append({
                        "score":  beam["score"] + (-lp_i),
                        "tokens": beam["tokens"] + [tok_i],
                        "hidden": h_new,
                    })

            if not candidates:
                break

            candidates.sort(key=lambda x: x["score"])
            beams = candidates[:beam_size]

            # Early stop if all beams ended
            if all(b["tokens"][-1] == vocab.end_idx for b in beams):
                for b in beams:
                    completed.append((b["score"], b["tokens"][:-1]))
                beams = []
                break

        # ── Collect remaining open beams ──────────────────────────────
        for b in beams:
            tokens = b["tokens"]
            if tokens and tokens[-1] == vocab.end_idx:
                tokens = tokens[:-1]
            completed.append((b["score"], tokens))

        if not completed:
            return []

        # Best = lowest cumulative negative log-probability
        best = min(completed, key=lambda x: x[0])
        return best[1]
