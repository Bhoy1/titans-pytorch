from titans_pytorch.associative_scan import AssocScan
import torch

# Initialize associative scan
assoc_scan = AssocScan(use_accelerated=False)

# Generate input tensors
batch_size = 1
seq_length = 256
feature_dim = 512

gates = torch.rand(batch_size, seq_length, feature_dim)  # Random gates
inputs = torch.randn(batch_size, seq_length, feature_dim)  # Random input sequences

# Forward pass
out = assoc_scan(gates, inputs)

print(f"AssocScan Output Shape: {out.shape}")  # Should be (1, 256, 512)
