from titans_pytorch.memory_models import MemoryMLP
import torch

# Initialize the MLP memory model
memory_mlp = MemoryMLP(
    dim=512,
    depth=3,
    expansion_factor=4.0
)

# Create a random input
x = torch.randn(1, 128, 512)  # (batch, sequence length, feature dimension)

# Forward pass
out = memory_mlp(x)

print(f"MemoryMLP Output Shape: {out.shape}")
