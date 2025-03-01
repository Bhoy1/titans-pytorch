from titans_pytorch import NeuralMemory
import torch

memory = NeuralMemory(
    dim=512,
    chunk_size=8,
)

x = torch.randn(1, 1024, 512)  # Batch of sequences
out, mem = memory(x)
print(out.shape)  # Should match input shape