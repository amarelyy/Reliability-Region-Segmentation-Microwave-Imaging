"""
unet/training/train.py

Main training loop for the Microwave Imaging U-Net.
"""

import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from unet.dataset.microwave_dataset import MicrowaveImagingDataset
from unet.models.unet_2ch import UNet2Ch
from unet.training.losses import CombinedLoss
from src.data_loading import load_all_data

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading UM-BMID data...")
    d = load_all_data()
    s21 = d["s21"]
    tumor_model = d["tumor_model"]
    id_to_original_idx = d["id_to_original_idx"]
    
    # Define freq_axis for bandpass
    freq_axis = np.linspace(1e9, 8e9, 1001)

    # Split data (Simple 80/20 split for now)
    n_scans = len(tumor_model)
    train_indices = list(range(int(0.8 * n_scans)))
    val_indices = list(range(int(0.8 * n_scans), n_scans))

    train_dataset = MicrowaveImagingDataset(s21, tumor_model.iloc[train_indices].reset_index(drop=True), 
                                            id_to_original_idx, freq_axis)
    val_dataset = MicrowaveImagingDataset(s21, tumor_model.iloc[val_indices].reset_index(drop=True), 
                                          id_to_original_idx, freq_axis)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 3. Initialize Model & Optimizer
    model = UNet2Ch(n_channels=2, n_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CombinedLoss()

    # 4. Training Loop
    epochs = 20
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 5. Save Model
    save_path = Path(__file__).resolve().parent.parent / "checkpoints" / "unet_reliability.pth"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()