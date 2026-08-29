"""
unet/inference/evaluate.py

Calculates quantitative metrics for the trained U-Net model.
"""

import torch
import numpy as np
from scipy.ndimage import center_of_mass

def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice Similarity Coefficient."""
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    intersection = np.sum(pred_flat * gt_flat)
    return (2. * intersection) / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-6)

def calculate_com_error(pred_mask, gt_mask, axis_mm):
    """Calculate Euclidean distance between centers of mass."""
    # Find center of mass for prediction and ground truth
    com_pred = center_of_mass(pred_mask)
    com_gt = center_of_mass(gt_mask)
    
    # Convert pixel indices to mm using axis_mm
    pred_x, pred_y = axis_mm[int(com_pred[1])], axis_mm[int(com_pred[0])]
    gt_x, gt_y = axis_mm[int(com_gt[1])], axis_mm[int(com_gt[0])]
    
    return np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

def evaluate_model(model, dataloader, device, axis_mm):
    model.eval()
    dice_scores = []
    com_errors = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Threshold at 0.5 to get binary mask from reliability map
            preds = (outputs > 0.5).float()
            
            # Move to CPU for numpy operations
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(preds_np.shape[0]):
                p = preds_np[i, 0]
                t = targets_np[i, 0]
                
                if np.sum(t) > 0: # Only evaluate if there is a tumor
                    dice_scores.append(calculate_dice(p, t))
                    com_errors.append(calculate_com_error(p, t, axis_mm))
                    
    mean_dice = np.mean(dice_scores) if dice_scores else 0
    mean_com = np.mean(com_errors) if com_errors else 0
    
    return mean_dice, mean_com