"""
unet_reliability/dataset/microwave_dataset.py

Custom PyTorch Dataset for Microwave Imaging.
Generates 2-channel input (Magnitude + CF) and Binary Ground Truth Mask.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

# Add project root to path to import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

class MicrowaveImagingDataset(Dataset):
    def __init__(self, s21, tumor_model, id_to_original_idx, freq_axis, 
                 transform=None, use_cf=True, grid_size=128):
        """
        Args:
            s21: Complex S-parameter array.
            tumor_model: DataFrame containing scan metadata.
            id_to_original_idx: Mapping for calibration references.
            freq_axis: Frequency points for bandpass filtering.
            transform: Optional torchvision transforms (e.g., augmentation).
            use_cf: Whether to include Coherence Factor as 2nd channel.
            grid_size: Target output size for the image/mask (e.g., 128x128).
        """
        self.s21 = s21
        self.tumor_model = tumor_model.reset_index(drop=True)
        self.id_to_original_idx = id_to_original_idx
        self.freq_axis = freq_axis
        self.transform = transform
        self.use_cf = use_cf
        self.grid_size = grid_size
        
        # Pre-compute axis_mm range based on a sample to ensure consistency
        # In a real scenario, we might want to normalize coordinates globally
        self.sample_row = self.tumor_model.iloc[0]
        self.breast_radius_sample = float(self.sample_row["breast_radius_mm"])

    def __len__(self):
        return len(self.tumor_model)

    def _create_gt_mask(self, row, axis_mm):
        """Create a binary mask from tumor metadata."""
        gt_x = float(row["tumor_x_mm"])
        gt_y = float(row["tumor_y_mm"])
        gt_r = float(row["tumor_radius_mm"])
        
        gx, gy = np.meshgrid(axis_mm, axis_mm)
        dist = np.sqrt((gx - gt_x)**2 + (gy - gt_y)**2)
        mask = (dist <= gt_r).astype(np.float32)
        return mask

    def __getitem__(self, idx):
        row = self.tumor_model.iloc[idx]
        
        # 1. Run Physics Pipeline to get Image + CF
        # Note: We disable depth gain/TVSVD here to keep input raw-ish for AI to learn
        result = reconstruct_scan(
            idx, self.s21, self.tumor_model, self.id_to_original_idx, 
            freq_axis=self.freq_axis,
            beamformer="das", use_snellius=True, use_cf=self.use_cf,
            use_tvsvd=False, use_bandpass=True, use_depth_gain=False,
            return_diagnostics=True
        )
        
        img = result["diagnostics"]["image"]
        cf_map = result["diagnostics"]["cf_map"] if self.use_cf else None
        axis_mm = result["diagnostics"]["axis_mm"]
        
        # 2. Resize/Normalize to target grid size
        # Simple center-crop or resize logic would go here. 
        # For now, assuming pipeline output matches grid_size or is handled by transform
        
        # Normalize Magnitude to [0, 1]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Stack Channels: (2, H, W)
        if self.use_cf and cf_map is not None:
            cf_norm = (cf_map - cf_map.min()) / (cf_map.max() - cf_map.min() + 1e-8)
            input_data = np.stack([img_norm, cf_norm], axis=0)
        else:
            input_data = img_norm[np.newaxis, ...]

        # 3. Create Ground Truth Mask
        gt_mask = self._create_gt_mask(row, axis_mm)
        gt_mask = gt_mask[np.newaxis, ...] # (1, H, W)

        # 4. Convert to Torch Tensors
        input_tensor = torch.from_numpy(input_data).float()
        mask_tensor = torch.from_numpy(gt_mask).float()

        if self.transform:
            input_tensor = self.transform(input_tensor)
            mask_tensor = self.transform(mask_tensor)

        return input_tensor, mask_tensor