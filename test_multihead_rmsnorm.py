from titans_pytorch.neural_memory import MultiheadRMSNorm
import torch

# Initialize normalization module
norm = MultiheadRMSNorm(dim=512, heads=8)

# Generate input tensor
x = torch.randn(1, 128, 512)  # (batch, sequence length, feature dimension)

# Apply normalization
out = norm(x)

print(f"MultiheadRMSNorm Output Shape: {out.shape}")
