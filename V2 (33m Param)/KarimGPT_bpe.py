"""
Step 3: GPT Language Model — BPE + Dual GPU + Bigger Model

Changes from previous version:
  - n_embd=512, n_head=8, n_layer=8 → ~33M params (was 13M)
  - DataParallel for dual T4 GPUs on Kaggle
  - max_iters=10000
  - batch_size=192 (split across 2 GPUs)
  - autocast in estimate_loss
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
import os

# ===================== HYPERPARAMETERS =====================
batch_size = 192
block_size = 256
max_iters = 10000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.2
# ===========================================================

print(f"Using device: {device}")
print(f"GPUs available: {torch.cuda.device_count()}")
torch.manual_seed(1337)

# ===================== LOAD TOKENIZER =====================
TOKENIZER_PATH = "tokenizer.json"
assert os.path.exists(TOKENIZER_PATH), "Run train_tokenizer.py first!"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
print(f"Loaded BPE tokenizer — vocab_size: {vocab_size}")

def encode(text: str) -> list[int]:
    return tokenizer.encode(text).ids

def decode(ids: list[int]) -> str:
    return tokenizer.decode(ids)

# ===================== LOAD & TOKENIZE DATA =====================
DATA_CACHE = "data.pt"

if os.path.exists(DATA_CACHE):
    print(f"Loading cached tokenized data from '{DATA_CACHE}'...")
    data = torch.load(DATA_CACHE, weights_only=True)
else:
    print("Tokenizing input.txt in chunks...")
    all_ids = []
    chunk_size = 5_000_000
    with open('input.txt', 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            all_ids.extend(encode(chunk))
            print(f"  Tokenized {len(all_ids):,} tokens so far...")

    data = torch.tensor(all_ids, dtype=torch.long)
    del all_ids
    torch.save(data, DATA_CACHE)
    print(f"Tokenized and cached to '{DATA_CACHE}'")

print(f"Dataset: {len(data):,} tokens")

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ===================== DATA LOADING =====================
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
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
            losses[k] = loss.mean().item()
        out[split] = losses.mean()
    model.train()
    return out

# ===================== MODEL =====================

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

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
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

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
            if end_token_id is not None and idx_next.item() == end_token_id:
                break

        return idx

# ===================== INIT MODEL =====================
model = GPTLanguageModel()
model = model.to(device)
param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{param_count:.2f}M parameters")

# Wrap in DataParallel if multiple GPUs available
raw_model = model  # keep reference to unwrapped model
if torch.cuda.device_count() > 1:
    print(f"Wrapping model in DataParallel across {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# ===================== TRAIN OR LOAD =====================
MODEL_PATH = 'model_bpe.pt'

if os.path.exists(MODEL_PATH):
    print(f"Found saved weights at '{MODEL_PATH}'. Loading...")
    raw_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"No saved weights found. Training for {max_iters} iterations...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler()

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(xb, yb)

        loss = loss.mean()  # aggregate across GPUs

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Training complete! Saving to '{MODEL_PATH}'...")
    torch.save(raw_model.state_dict(), MODEL_PATH)

# ===================== GENERATION =====================
print("-" * 50)
raw_model.eval()

inp = input("Enter prompt: ")
prompt = f"<|user|>\n{inp}\n<|assistant|>\n"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

print("\nGenerating response:")
output_ids = raw_model.generate(context, max_new_tokens=200)[0].tolist()
print(decode(output_ids))