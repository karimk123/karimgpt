"""
GPT-2 Small (124M) — DDP + torch.compile + nanoGPT techniques.

Launch with torchrun (in a Kaggle/Colab notebook cell):
  !torchrun --standalone --nproc_per_node=2 gpt2_train_v2.py

Or single GPU:
  !python gpt2_train_v2.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tokenizers import Tokenizer
import os
import time
import math
import warnings
warnings.filterwarnings("ignore")

# ===================== HYPERPARAMETERS =====================
# Model
n_embd = 768
n_head = 12
n_layer = 12
block_size = 512
dropout = 0.0
bias = False

# Training
micro_batch_size = 16
gradient_accumulation_steps = 8
max_iters = 20000
eval_interval = 500
eval_iters = 100
checkpoint_interval = 2000

# Optimizer
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 1000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# System
use_compile = True
# ===========================================================

# ===================== DDP SETUP =====================
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)  # only rank 0 logs and saves
    # Scale down gradient accumulation since multiple GPUs share the work
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # Single GPU fallback
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ddp_world_size = 1
    master_process = True

if master_process:
    print(f"Using device: {device}")
    print(f"DDP: {ddp} | World size: {ddp_world_size}")

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

effective_batch_tokens = micro_batch_size * gradient_accumulation_steps * ddp_world_size * block_size
if master_process:
    print(f"Effective tokens per step: {effective_batch_tokens:,}")

# ===================== LOAD TOKENIZER =====================
TOKENIZER_PATH = "tokenizer.json"
assert os.path.exists(TOKENIZER_PATH), "Run train_tokenizer_v2.py first!"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
if master_process:
    print(f"Loaded BPE tokenizer — vocab_size: {vocab_size}")

def encode(text: str) -> list[int]:
    return tokenizer.encode(text).ids

def decode(ids: list[int]) -> str:
    return tokenizer.decode(ids)

# ===================== LOAD & TOKENIZE DATA =====================
DATA_CACHE = "data.pt"

if os.path.exists(DATA_CACHE):
    if master_process:
        print(f"Loading cached tokenized data from '{DATA_CACHE}'...")
    data = torch.load(DATA_CACHE, weights_only=True)
else:
    if master_process:
        print("Tokenizing input.txt in chunks...")
    all_ids = []
    chunk_size = 5_000_000
    with open('input.txt', 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            all_ids.extend(encode(chunk))
            if master_process:
                print(f"  Tokenized {len(all_ids):,} tokens so far...")

    data = torch.tensor(all_ids, dtype=torch.long)
    del all_ids
    torch.save(data, DATA_CACHE)

if master_process:
    print(f"Dataset: {len(data):,} tokens")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ===================== DATA LOADING =====================
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (micro_batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ===================== LEARNING RATE SCHEDULE =====================
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ===================== MODEL =====================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if self.flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            tril = torch.tril(torch.ones(T, T, device=x.device))
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(tril == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.attn_dropout(wei)
            out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.ln2 = nn.LayerNorm(n_embd, bias=bias)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            end_token_id = tokenizer.token_to_id("<|end|>")
            user_token_id = tokenizer.token_to_id("<|user|>")
            if end_token_id is not None and idx_next.item() == end_token_id:
                break
            if user_token_id is not None and idx_next.item() == user_token_id:
                break

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (nn.LayerNorm, nn.Embedding)):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, nn.Linear):
                    decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        decay.discard('lm_head.weight')  # weight-tied, already in no_decay via Embedding

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        if master_process:
            num_decay = sum(param_dict[pn].numel() for pn in decay)
            num_no_decay = sum(param_dict[pn].numel() for pn in no_decay)
            print(f"Decay params: {num_decay:,} | No-decay params: {num_no_decay:,}")

        use_fused = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                       fused=use_fused if device.startswith('cuda') else False)
        return optimizer

# ===================== INIT MODEL =====================
model = GPTLanguageModel()
model = model.to(device)

if master_process:
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{param_count:.2f}M parameters")

# Compile BEFORE DDP wrapping
if use_compile:
    if master_process:
        print("Compiling model with torch.compile (takes ~1 min)...")
    model = torch.compile(model)

# Store reference to raw model for saving/loading/generating
raw_model = model

# Wrap in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module  # unwrap DDP to access the actual model

# ===================== CHECKPOINT SYSTEM =====================
CHECKPOINT_PATH = 'checkpoint.pt'
MODEL_PATH = 'model_gpt2.pt'

start_iter = 0
best_val_loss = 1e9

if os.path.exists(CHECKPOINT_PATH):
    if master_process:
        print("Found checkpoint, resuming...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    raw_model.load_state_dict(checkpoint['model_state_dict'])
    start_iter = checkpoint['iter'] + 1
    best_val_loss = checkpoint.get('best_val_loss', 1e9)
    if master_process:
        print(f"Resuming from step {start_iter}")
elif os.path.exists(MODEL_PATH):
    if master_process:
        print("Found finished model. Loading for inference...")
    raw_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    start_iter = max_iters

# ===================== TRAINING =====================
if start_iter < max_iters:
    if master_process:
        print(f"Training from step {start_iter} to {max_iters}...")

    optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))
    scaler = torch.amp.GradScaler()

    if os.path.exists(CHECKPOINT_PATH) and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if master_process:
            print("Restored optimizer and scaler state")

    start_time = time.time()

    for iter in range(start_iter, max_iters):

        # Set learning rate
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate and log (master only)
        if iter % eval_interval == 0 or iter == max_iters - 1:
            if master_process:
                losses = estimate_loss()
                elapsed = time.time() - start_time
                steps_done = max(iter - start_iter, 1)
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
                eta_hours = ((max_iters - iter) / steps_per_sec / 3600) if steps_per_sec > 0 else 0
                print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f} | "
                      f"lr {lr:.2e} | {steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h")

                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']

        # --- Gradient accumulation loop ---
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(gradient_accumulation_steps):
            xb, yb = get_batch('train')

            # Only sync gradients on the last micro step (DDP optimization)
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(xb, yb)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

        # Gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Save checkpoint (master only)
        if master_process and (iter + 1) % checkpoint_interval == 0:
            print(f"Saving checkpoint at step {iter}...")
            torch.save({
                'iter': iter,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
            }, CHECKPOINT_PATH)

    if master_process:
        total_time = time.time() - start_time
        print(f"Training complete in {total_time/3600:.1f} hours!")
        print(f"Saving final model to '{MODEL_PATH}'...")
        torch.save(raw_model.state_dict(), MODEL_PATH)

# ===================== GENERATION (master only) =====================
if master_process:
    print("-" * 50)
    raw_model.eval()

    try:
        inp = input("Enter prompt: ")
        prompt = f"<|user|>\n{inp}\n<|assistant|>\n"
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

        print("\nGenerating response:")
        output_ids = raw_model.generate(context, max_new_tokens=300)[0].tolist()
        print(decode(output_ids))
    except EOFError:
        print("Non-interactive mode — generating samples...")
        prompts = [
            "What is the capital of France?",
            "How do I make pasta?",
            "Explain what a neural network is.",
            "Write a short poem about the ocean.",
            "What are the benefits of exercise?",
        ]
        for p in prompts:
            prompt = f"<|user|>\n{p}\n<|assistant|>\n"
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            output_ids = raw_model.generate(context, max_new_tokens=300)[0].tolist()
            print(f"\n{'='*60}")
            print(decode(output_ids))

# Cleanup DDP
if ddp:
    destroy_process_group()