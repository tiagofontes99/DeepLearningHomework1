import torch
from torch.utils.data import DataLoader
from utils import load_rnacompete_data
import utils

# Set seed for reproducibility
utils.configure_seed(42)

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load Data for a specific protein (e.g., 'RBFOX1', 'PTB', 'A1CF')
# This returns a PyTorch TensorDataset ready for training
train_dataset = load_rnacompete_data(protein_name='RBFOX1', split='train')
val_dataset   = load_rnacompete_data(protein_name='RBFOX1', split='val')
test_dataset  = load_rnacompete_data(protein_name='RBFOX1', split='test')

# Wrap in a standard PyTorch DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training Loop Example
for batch in train_loader:
    # Unpack the batch: (Sequences, Intensities, ValidityMasks)
    x, y, mask = batch
    
    # x shape:    (Batch, 41, 4)  <- One-Hot Encoded Sequence
    # y shape:    (Batch, 1)      <- Normalized Binding Intensity
    # mask shape: (Batch, 1)      <- 1.0 if valid, 0.0 if NaN
    
    # Forward pass
    predictions = model(x)
    
    # Calculate Loss - use the mask to zero out invalid data points
    loss = utils.masked_mse_loss(predictions, y, mask)