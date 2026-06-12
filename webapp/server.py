"""
KarimGPT Web Backend
====================
A FastAPI server that serves ALL THREE KarimGPT models behind a single,
ChatGPT-style streaming API:

  - KarimGPT 1 — 13M  char-level  (../V1 (13m Param)/model.pt)
  - KarimGPT 2 — 33M  BPE chat    (../V2 (33m Param)/model_bpe.pt)
  - KarimGPT 3 — 91M  GPT-2 style (../V3 (91m Param)/model_gpt2.pt)

This file is self-contained on purpose: it re-declares the model
architectures used by the training scripts (gpt.py, KarimGPT_bpe.py,
gpt2_train.py) so the original training code is left completely untouched.
Weights and tokenizers are loaded BY REFERENCE from the version folders —
nothing is copied or modified.

Models are loaded lazily on first request (and cached), so the server can
start instantly and only pay the RAM cost of models that are actually used.
Set KARIMGPT_PRELOAD="v1,v2,v3" to eagerly load some/all at startup instead.

Run locally:
    pip install -r requirements.txt
    python server.py
Then open http://localhost:8000 — or deploy public/ to Vercel and point it
at this server's URL with ?api=https://your-backend.
"""

import os
import json
import asyncio
import threading
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===================== PATHS =====================
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the KarimGPT folder containing V1/V2/V3

V1_DIR = os.environ.get("KARIMGPT_V1_DIR", os.path.join(ROOT, "V1 (13m Param)"))
V2_DIR = os.environ.get("KARIMGPT_V2_DIR", os.path.join(ROOT, "V2 (33m Param)"))
V3_DIR = os.environ.get("KARIMGPT_V3_DIR", os.path.join(ROOT, "V3 (91m Param)"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== TOKENIZERS =====================
class CharTokenizer:
    """Mirror of V1's character-level vocab, rebuilt from the training corpus."""

    def __init__(self, corpus_path: str):
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def get_vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        # Drop characters the model has never seen (its vocab is just the corpus).
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def token_to_id(self, token: str):
        return None  # char model has no special tokens


class BPETokenizer:
    """Thin wrapper around a HuggingFace tokenizers file."""

    def __init__(self, path: str):
        self.tk = Tokenizer.from_file(path)

    def get_vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids)

    def token_to_id(self, token: str):
        return self.tk.token_to_id(token)


# ===================== MODEL SPECS =====================
@dataclass
class ModelSpec:
    id: str
    name: str
    params: str                 # human label, e.g. "33M"
    description: str
    dir: str
    weights_file: str
    tokenizer_kind: str         # "char" | "bpe"
    tokenizer_file: str         # corpus for char, tokenizer.json for bpe
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    bias: bool                  # Linear/LayerNorm bias (V3 trains with bias=False)
    activation: str             # "relu" | "gelu"
    weight_tying: bool
    flash: bool                 # V3 uses scaled_dot_product_attention (no tril buffer)
    prompt_style: str           # "chat" | "completion"
    stop_tokens: list = field(default_factory=list)
    temperature: float = 0.8
    top_k: int | None = 40
    max_new_tokens: int = 200


SPECS: dict[str, ModelSpec] = {
    "v1": ModelSpec(
        id="v1", name="KarimGPT 1", params="13M",
        description="Character-level · writes Shakespeare",
        dir=V1_DIR, weights_file="model.pt",
        tokenizer_kind="char", tokenizer_file="shakespear.txt",
        block_size=256, n_embd=384, n_head=6, n_layer=6,
        bias=True, activation="relu", weight_tying=False, flash=False,
        prompt_style="completion", stop_tokens=[],
        temperature=1.0, top_k=None, max_new_tokens=500,
    ),
    "v2": ModelSpec(
        id="v2", name="KarimGPT 2", params="33M",
        description="BPE · trained on 50k conversations",
        dir=V2_DIR, weights_file="model_bpe.pt",
        tokenizer_kind="bpe", tokenizer_file="tokenizer.json",
        block_size=256, n_embd=512, n_head=8, n_layer=8,
        bias=True, activation="relu", weight_tying=False, flash=False,
        prompt_style="chat", stop_tokens=["<|end|>"],
        temperature=0.8, top_k=40, max_new_tokens=200,
    ),
    "v3": ModelSpec(
        id="v3", name="KarimGPT 3", params="91M",
        description="GPT-2 style · largest and most capable",
        dir=V3_DIR, weights_file="model_gpt2.pt",
        tokenizer_kind="bpe", tokenizer_file="tokenizer.json",
        block_size=512, n_embd=768, n_head=12, n_layer=12,
        bias=False, activation="gelu", weight_tying=True, flash=True,
        prompt_style="chat", stop_tokens=["<|end|>", "<|user|>"],
        temperature=0.8, top_k=40, max_new_tokens=300,
    ),
}

DEFAULT_MODEL = "v3"


# ===================== ARCHITECTURE (mirrors the training scripts) =====================
class Head(nn.Module):
    def __init__(self, spec: ModelSpec, head_size: int):
        super().__init__()
        self.spec = spec
        # k/q/v are bias-free in all three training scripts.
        self.key = nn.Linear(spec.n_embd, head_size, bias=False)
        self.query = nn.Linear(spec.n_embd, head_size, bias=False)
        self.value = nn.Linear(spec.n_embd, head_size, bias=False)
        if not spec.flash:
            # V1/V2 keep the causal mask as a persistent buffer (present in the
            # saved state dicts), so it must be registered here too.
            self.register_buffer("tril", torch.tril(torch.ones(spec.block_size, spec.block_size)))
        self.dropout = nn.Dropout(0.0)  # eval only

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        if self.spec.flash and hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        if self.spec.flash:
            tril = torch.tril(torch.ones(T, T, device=x.device))
            mask = tril == 0
        else:
            mask = self.tril[:T, :T] == 0
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(mask, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, spec: ModelSpec, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(spec, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, spec.n_embd, bias=spec.bias)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        act = nn.GELU() if spec.activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(spec.n_embd, 4 * spec.n_embd, bias=spec.bias),
            act,
            nn.Linear(4 * spec.n_embd, spec.n_embd, bias=spec.bias),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


def _layer_norm(spec: ModelSpec):
    return nn.LayerNorm(spec.n_embd) if spec.bias else nn.LayerNorm(spec.n_embd, bias=False)


class Block(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        head_size = spec.n_embd // spec.n_head
        self.sa = MultiHeadAttention(spec, spec.n_head, head_size)
        self.ffwd = FeedFoward(spec)
        self.ln1 = _layer_norm(spec)
        self.ln2 = _layer_norm(spec)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, spec: ModelSpec, vocab_size: int):
        super().__init__()
        self.spec = spec
        self.token_embedding_table = nn.Embedding(vocab_size, spec.n_embd)
        self.position_embedding_table = nn.Embedding(spec.block_size, spec.n_embd)
        self.blocks = nn.Sequential(*[Block(spec) for _ in range(spec.n_layer)])
        self.ln_f = _layer_norm(spec)
        self.lm_head = nn.Linear(spec.n_embd, vocab_size, bias=False if spec.weight_tying else spec.bias)
        if spec.weight_tying:
            self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def stream(self, idx, max_new_tokens, temperature, top_k, stop_ids):
        """Yield one new token id at a time (same sampling as the training scripts)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.spec.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            tok = idx_next.item()
            if tok in stop_ids:
                break
            yield tok


# ===================== LAZY MODEL REGISTRY =====================
class LoadedModel:
    def __init__(self, spec: ModelSpec, tokenizer, model: GPTLanguageModel):
        self.spec = spec
        self.tokenizer = tokenizer
        self.model = model
        self.stop_ids = set()
        for t in spec.stop_tokens:
            tid = tokenizer.token_to_id(t)
            if tid is not None:
                self.stop_ids.add(tid)
        self.n_params = sum(p.numel() for p in model.parameters()) / 1e6


_loaded: dict[str, LoadedModel] = {}
_locks: dict[str, threading.Lock] = {mid: threading.Lock() for mid in SPECS}


def _clean_state_dict(sd):
    """Normalize a checkpoint into a plain state dict the mirror class accepts."""
    if isinstance(sd, nn.Module):
        sd = sd.state_dict()
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    for prefix in ("_orig_mod.", "module."):  # torch.compile / DataParallel wrappers
        if any(k.startswith(prefix) for k in sd):
            sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
    return sd


def get_model(model_id: str) -> LoadedModel:
    """Load a model on first use (thread-safe), then serve it from cache."""
    if model_id in _loaded:
        return _loaded[model_id]

    spec = SPECS[model_id]
    with _locks[model_id]:
        if model_id in _loaded:  # another request loaded it while we waited
            return _loaded[model_id]

        tok_path = os.path.join(spec.dir, spec.tokenizer_file)
        weights_path = os.path.join(spec.dir, spec.weights_file)
        assert os.path.exists(tok_path), f"{spec.tokenizer_file} not found at {tok_path}"
        assert os.path.exists(weights_path), f"{spec.weights_file} not found at {weights_path}"

        print(f"[{spec.name}] loading tokenizer: {tok_path}")
        tokenizer = CharTokenizer(tok_path) if spec.tokenizer_kind == "char" else BPETokenizer(tok_path)

        print(f"[{spec.name}] loading weights: {weights_path}")
        model = GPTLanguageModel(spec, tokenizer.get_vocab_size()).to(DEVICE)
        try:
            sd = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        except Exception as e:
            # weights_only=False unpickles arbitrary objects (code execution risk),
            # so it must be explicitly opted into for checkpoints saved as full
            # modules. These are your own training outputs, but stay safe by default.
            if os.environ.get("KARIMGPT_ALLOW_UNSAFE_LOAD") != "1":
                raise RuntimeError(
                    f"{spec.weights_file} is not a plain tensor state dict ({e}). "
                    "If you trust this file (you trained it), set "
                    "KARIMGPT_ALLOW_UNSAFE_LOAD=1 and restart."
                ) from e
            sd = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(_clean_state_dict(sd))
        model.eval()

        lm = LoadedModel(spec, tokenizer, model)
        _loaded[model_id] = lm
        print(f"[{spec.name}] ready — {lm.n_params:.2f}M params, vocab {tokenizer.get_vocab_size()}, device {DEVICE}")
        return lm


# ===================== PROMPTING =====================
def build_prompt(spec: ModelSpec, messages: list) -> str:
    if spec.prompt_style == "chat":
        # Training format per turn: <|user|>\n{text}\n<|assistant|>\n{reply}<|end|>
        parts = []
        for m in messages:
            tag = "<|user|>" if m.role == "user" else "<|assistant|>"
            parts.append(f"{tag}\n{m.content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)
    # Completion model (V1): continue the conversation as raw text.
    return "\n".join(m.content for m in messages) + "\n"


# ===================== API =====================
app = FastAPI(title="KarimGPT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = DEFAULT_MODEL
    temperature: float | None = None
    top_k: int | None = None
    max_new_tokens: int | None = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "loaded": sorted(_loaded.keys()),
    }


@app.get("/api/models")
def list_models():
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {
                "id": s.id,
                "name": s.name,
                "params": s.params,
                "description": s.description,
                "style": s.prompt_style,
                "loaded": s.id in _loaded,
            }
            for s in SPECS.values()
        ],
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    model_id = req.model if req.model in SPECS else DEFAULT_MODEL
    spec = SPECS[model_id]

    async def event_stream():
        loop = asyncio.get_event_loop()

        # Load (or fetch cached) model off the event loop; tell the UI if it's a cold start.
        if model_id not in _loaded:
            yield f"data: {json.dumps({'status': 'loading', 'model': spec.name})}\n\n"
        try:
            lm = await loop.run_in_executor(None, get_model, model_id)
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Failed to load {spec.name}: {e}'})}\n\n"
            return

        prompt = build_prompt(spec, req.messages)
        ids = lm.tokenizer.encode(prompt)
        if not ids:
            ids = lm.tokenizer.encode("\n") or [0]
        # Keep the most recent context within the model's window.
        ids = ids[-spec.block_size:]
        context = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        gen = lm.model.stream(
            context,
            max_new_tokens=int(req.max_new_tokens or spec.max_new_tokens),
            temperature=float(req.temperature if req.temperature is not None else spec.temperature),
            top_k=int(req.top_k) if req.top_k is not None else spec.top_k,
            stop_ids=lm.stop_ids,
        )

        emitted = []    # token ids emitted so far
        prev_text = ""  # decoded text already sent to client

        while True:
            tok = await loop.run_in_executor(None, lambda: next(gen, None))
            if tok is None:
                break
            emitted.append(tok)
            # Decode the whole emitted sequence and send only the new delta.
            # This keeps multi-token BPE words joined correctly.
            text = lm.tokenizer.decode(emitted)
            delta = text[len(prev_text):]
            prev_text = text
            if delta:
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            await asyncio.sleep(0)  # let the event loop flush

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Optional eager loading, e.g. KARIMGPT_PRELOAD="v2,v3"
_preload = os.environ.get("KARIMGPT_PRELOAD", "").strip()
if _preload:
    for mid in [m.strip() for m in _preload.split(",") if m.strip() in SPECS]:
        get_model(mid)


# Serve the static frontend too (handy for local single-process use).
try:
    from fastapi.staticfiles import StaticFiles
    public_dir = os.path.join(HERE, "public")
    if os.path.isdir(public_dir):
        app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")
except Exception as e:  # pragma: no cover
    print(f"(static mount skipped: {e})")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
