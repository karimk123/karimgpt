"""
Step 2: Train BPE tokenizer with larger vocab for the 124M model.
vocab_size=8192 — bigger model can handle the larger embedding table.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# --- CONFIG ---
INPUT_FILE = "input.txt"
VOCAB_SIZE = 8192
TOKENIZER_PATH = "tokenizer.json"
# ---------------

print(f"Training BPE tokenizer with vocab_size={VOCAB_SIZE}...")

tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

special_tokens = ["<|unk|>", "<|pad|>", "<|user|>", "<|assistant|>", "<|end|>"]

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=special_tokens,
    min_frequency=2,
    show_progress=True,
)

tokenizer.train([INPUT_FILE], trainer)
tokenizer.save(TOKENIZER_PATH)
print(f"Tokenizer saved to '{TOKENIZER_PATH}'")

# Quick test
print("\n--- Tokenizer Test ---")
test_text = "<|user|>\nWhat is the capital of France?\n<|assistant|>\nThe capital of France is Paris.\n<|end|>"
encoded = tokenizer.encode(test_text)
print(f"Vocab size: {tokenizer.get_vocab_size()}")
print(f"Token count: {len(encoded.ids)} tokens (was {len(test_text)} characters)")
print(f"Compression: {len(test_text) / len(encoded.ids):.1f}x")

for tok in special_tokens:
    print(f"  {tok} → {tokenizer.token_to_id(tok)}")