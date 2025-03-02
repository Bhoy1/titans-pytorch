import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import gzip
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP, MemoryAttention

# Constants
BATCH_SIZE = 4
SEQ_LEN = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
DIM = 128  # Ensure consistency between models

# Neural Memory Hyperparameters
NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4)  # Apply memory at transformer layers 2 and 4
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True

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

# Choose Memory Model
USE_MEM_ATTENTION_MODEL = False  # Set to True to use MemoryAttention

if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(dim=DIM)
else:
    neural_memory_model = MemoryMLP(dim=DIM, depth=NEURAL_MEMORY_DEPTH)

# Initialize model with enhanced features
model = MemoryAsContextTransformer(
    num_tokens=256,
    dim=DIM,  
    depth=4,  
    segment_len=32,  
    num_persist_mem_tokens=NUM_PERSIST_MEM,  
    num_longterm_mem_tokens=NUM_LONGTERM_MEM,  
    neural_memory_layers=NEURAL_MEM_LAYERS,  
    neural_memory_model=neural_memory_model,
    sliding_window_attn=SLIDING_WINDOWS,
    neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTN_OUTPUT,
    neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    neural_memory_kwargs={
        "dim_head": 64,
        "heads": 4,
        "attn_pool_chunks": STORE_ATTN_POOL_CHUNKS,
        "qk_rmsnorm": NEURAL_MEM_QK_NORM,
        "momentum": NEURAL_MEM_MOMENTUM,
        "momentum_order": NEURAL_MEM_MOMENTUM_ORDER,
        "default_step_transform_max_lr": NEURAL_MEM_MAX_LR,
        "per_parameter_lr_modulation": MEMORY_MODEL_PER_LAYER_LEARNED_LR
    }
)

# Move to GPU if available
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
    print(f"Starting epoch {epoch+1}/{NUM_EPOCHS}")  

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move batch to GPU
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]

        loss = model(inputs, return_loss=True)  # Already returns loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")  

        # Validate every 100 batches
        if batch_idx % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_loss = model(inputs, return_loss=True)
                print(f"Validation Loss at Batch {batch_idx}: {val_loss.item()}")
            model.train()

        # Generate text every 500 batches
        if batch_idx % GENERATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                sample_input = batch[0, :PRIME_LENGTH].unsqueeze(0)  # Take the first sequence as prime
                generated = model.sample(sample_input, GENERATE_LENGTH)
                print(f"Generated Text:\n{generated}")
            model.train()
