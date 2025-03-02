import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import gzip
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# Constants
BATCH_SIZE = 4
SEQ_LEN = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
DIM = 128  # Ensure consistency between models

# Load Data
def load_enwik8():
    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    return torch.from_numpy(data).long()

# Simple Dataset
class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) // self.seq_len
    
    def __getitem__(self, idx):
        start = torch.randint(0, len(self.data) - self.seq_len - 1, (1,)).item()
        return self.data[start : start + self.seq_len + 1]

# Initialize model with matching dimensions
memory_model = MemoryMLP(dim=DIM, depth=2)
model = MemoryAsContextTransformer(
    num_tokens=256,
    dim=DIM,  # Now consistent
    depth=4,
    segment_len=32,
    num_persist_mem_tokens=4,
    num_longterm_mem_tokens=4,
    neural_memory_model=memory_model
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

# Prepare data
data = load_enwik8()
dataset = TextDataset(data, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"Starting epoch {epoch+1}/{NUM_EPOCHS}")  # NEW: Print epoch start
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move batch to GPU before passing to model
        batch = batch.to(device)  # NEW: Move the whole batch to GPU
        inputs, targets = batch[:, :-1], batch[:, 1:]  # Extract inputs & targets

        loss = model(inputs, return_loss=True)  # Already returns loss
        loss.backward()
        
        optimizer.step()

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")  # NEW: Show batch loss


