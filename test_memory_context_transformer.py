from titans_pytorch import MemoryAsContextTransformer
import torch

# Initialize the transformer
transformer = MemoryAsContextTransformer(
    num_tokens=20000,   # Vocabulary size
    dim=512,            # Feature dimension
    depth=6,            # Number of layers
    segment_len=256,    # Length of input segment
    num_persist_mem_tokens=16,  # Persistent memory tokens
    num_longterm_mem_tokens=64  # Long-term memory tokens
)

# Simulate input token IDs
tokens = torch.randint(0, 20000, (1, 256))  # (batch, sequence length)

# Compute loss
loss = transformer(tokens, return_loss=True)
loss.backward()

print(f"Transformer loss: {loss.item()}")
