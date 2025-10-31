# File: app/core/xai/visualize_gradcam_2d.py
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import os
import cv2 # Needed for resizing
import matplotlib.pyplot as plt # For plotting
import matplotlib.cm as cm # For colormaps
from typing import List, Dict, Any, Tuple, Optional 

# (Import constants and preprocessing function from model script)
try:
    # We need the preprocessing function to get the original slices
    # and constants for validation
    from model.shufflenet.model import preprocess_nii_to_slices, NUM_SLICES_PER_SUBJECT, SLICE_IMG_SIZE 
except ImportError as e:
    print(f"Error importing from model.shufflenet.model: {e}")
    # Provide defaults and a placeholder function if import fails
    NUM_SLICES_PER_SUBJECT = 10
    SLICE_IMG_SIZE = 128
    def preprocess_nii_to_slices(path):
        print("Error: Real preprocess_nii_to_slices function not found.")
        return None 
    print("Warning: Using default constants and placeholder preprocess function.")

# --- Grad-CAM Calculation Function (Copied from act_to_nii_gradcam.py) ---
def calculate_gradcam(activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Calculates Grad-CAM heatmap."""
    # ... (Exact same implementation as before) ...
    if activation.shape != gradient.shape: raise ValueError("Shape mismatch")
    n_slices, n_channels, height, width = activation.shape
    alpha = torch.mean(gradient, dim=(2, 3)).view(n_slices, n_channels, 1, 1)
    weighted_activations = alpha * activation
    cam = torch.sum(weighted_activations, dim=1)
    cam = F.relu(cam)
    cam_flat = cam.view(n_slices, -1)
    cam_min = torch.min(cam_flat, dim=1, keepdim=True)[0].view(n_slices, 1, 1)
    cam_max = torch.max(cam_flat, dim=1, keepdim=True)[0].view(n_slices, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8) 
    return cam # Shape [10, h, w]

# --- Main Visualization Function ---
def visualize_gradcam_2d(
    data_path: str, # Path to .pt dict {'activation': ..., 'gradient': ...}
    reference_nii_path: str, # Path to original T1 NIfTI
    output_dir: str, # Directory to save PNG images
    colormap: str = 'jet', # Colormap for the heatmap (e.g., 'jet', 'hot')
    alpha: float = 0.5 # Transparency of the heatmap overlay
):
    """
    Generates and saves 2D Grad-CAM visualizations overlaid on original slices.
    """
    print("--- Starting 2D Grad-CAM Visualization ---")
    
    # --- Step 1: Load Activation and Gradient ---
    try:
        data_dict = torch.load(data_path, map_location=torch.device('cpu')) 
        activation = data_dict['activation']
        gradient = data_dict['gradient']
        print(f"Loaded activation (shape: {activation.shape}) and gradient (shape: {gradient.shape})")
    except Exception as e:
        print(f"Error loading data file {data_path}: {e}")
        return False
        
    # Validate shapes
    if activation.ndim != 4 or activation.shape[0] != NUM_SLICES_PER_SUBJECT or \
       gradient.ndim != 4 or gradient.shape[0] != NUM_SLICES_PER_SUBJECT or \
       activation.shape[1:] != gradient.shape[1:]:
        print("Error: Invalid activation or gradient shape.")
        return False
        
    # --- Step 2: Calculate Grad-CAM Heatmap ---
    print("Calculating Grad-CAM...")
    with torch.no_grad():
      heatmap_small = calculate_gradcam(activation, gradient) # Shape [10, h, w]
      print(f"Calculated small heatmap shape: {heatmap_small.shape}")

    # --- Step 3: Upsample Heatmap to Target Size (128x128) ---
    heatmap_small_np = heatmap_small.numpy()
    target_size = (SLICE_IMG_SIZE, SLICE_IMG_SIZE) # (width, height for cv2)
    upsampled_heatmaps = []
    print(f"Upsampling heatmaps to {target_size}...")
    for i in range(NUM_SLICES_PER_SUBJECT):
        # Use INTER_LINEAR for smoother results
        resized = cv2.resize(heatmap_small_np[i], target_size, 
                             interpolation=cv2.INTER_LINEAR) 
        upsampled_heatmaps.append(resized)
    # Stack back: shape [10, 128, 128]
    upsampled_heatmaps_np = np.stack(upsampled_heatmaps)
    print(f"Upsampled heatmaps shape: {upsampled_heatmaps_np.shape}")

    # --- Step 4: Get Original Input Slices ---
    print(f"Loading original slices from: {reference_nii_path}")
    # preprocess_nii_to_slices returns numpy array [10, 1, 128, 128]
    original_slices_array = preprocess_nii_to_slices(reference_nii_path)
    if original_slices_array is None:
        print("Error: Failed to preprocess reference NIfTI to get original slices.")
        return False
    # Remove channel dimension: [10, 128, 128]
    original_slices_np = original_slices_array.squeeze(axis=1) 
    print(f"Loaded original slices shape: {original_slices_np.shape}")
    
    # --- Step 5: Plot Overlay and Save ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {NUM_SLICES_PER_SUBJECT} overlay images to: {output_dir}")
    
    saved_paths = []
    for i in range(NUM_SLICES_PER_SUBJECT):
        slice_img = original_slices_np[i]
        heatmap_img = upsampled_heatmaps_np[i]
        
        plt.figure(figsize=(6, 6)) # Adjust figure size as needed
        plt.imshow(slice_img, cmap='gray') # Show background slice
        plt.imshow(heatmap_img, cmap=colormap, alpha=alpha) # Overlay heatmap
        plt.title(f"Grad-CAM Overlay (Slice {i+1}/{NUM_SLICES_PER_SUBJECT})")
        plt.axis('off') # Hide axes
        
        output_filename = os.path.join(output_dir, f"slice_{i+1:02d}_gradcam.png")
        try:
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
            saved_paths.append(output_filename)
        except Exception as e:
             print(f"Error saving figure {output_filename}: {e}")
        plt.close() # Close the figure to free memory

    if len(saved_paths) == NUM_SLICES_PER_SUBJECT:
        print("All overlay images saved successfully.")
        return True
    else:
        print(f"Warning: Only {len(saved_paths)}/{NUM_SLICES_PER_SUBJECT} images were saved.")
        return False


if __name__ == "__main__":
    
    # --- Configuration for the Test ---
    # Path to where activation AND gradient are saved (as a dictionary)
    DATA_PATH = "output/papermodel_test/act_and_grad_backbone.stage4.1.gconv2.pt" 
    
    # Path to the ORIGINAL T1 NIfTI file used as input to the model
    REF_PATH = "/Volumes/3T-disk/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii" 
    
    # Directory to save the output PNG files
    OUT_DIR = "output/papermodel_test/gradcam_2d_visualizations" 

    # --- Mock data generation (if .pt file doesn't exist) ---
    if not os.path.exists(os.path.dirname(DATA_PATH)): os.makedirs(os.path.dirname(DATA_PATH))
    if not os.path.exists(DATA_PATH):
        print(f"Mocking activation and gradient file at: {DATA_PATH}")
        mock_act = torch.randn(NUM_SLICES_PER_SUBJECT, 960, 4, 4) 
        mock_grad = torch.abs(torch.randn(NUM_SLICES_PER_SUBJECT, 960, 4, 4)) 
        mock_data = {'activation': mock_act, 'gradient': mock_grad}
        torch.save(mock_data, DATA_PATH)
    # --- End Mock ---

    # --- Run the main function ---
    if not os.path.exists(REF_PATH):
        print(f"Error: Reference NIfTI file not found at {REF_PATH}")
    elif not os.path.exists(DATA_PATH):
         print(f"Error: Activation/Gradient data file not found at {DATA_PATH}")
    else:
        success = visualize_gradcam_2d(
            data_path=DATA_PATH,
            reference_nii_path=REF_PATH,
            output_dir=OUT_DIR,
        )
        if success:
             print("\n--- 2D Visualization Script finished successfully ---")
        else:
             print("\n--- 2D Visualization Script finished with errors ---")