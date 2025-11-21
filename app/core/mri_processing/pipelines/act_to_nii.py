import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import os
import cv2 # Needed for resizing
from typing import List, Dict, Any, Tuple, Optional 

# (Import constants from model script)
# Constants for slice processing
NUM_SLICES_PER_SUBJECT = 10
SLICE_IMG_SIZE = 128

# --- NEW: Grad-CAM Calculation Function ---
def calculate_gradcam(activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """
    Calculates Grad-CAM heatmap.
    
    Args:
        activation (torch.Tensor): Activation tensor from the target layer, 
                                   shape [N_slices, C, h, w] (e.g., [10, 960, 4, 4])
        gradient (torch.Tensor): Gradient tensor w.r.t. the target class score, 
                                 backpropagated to the target layer, 
                                 shape [N_slices, C, h, w]
                                 
    Returns:
        torch.Tensor: Grad-CAM heatmap, shape [N_slices, h, w] (e.g., [10, 4, 4])
    """
    if activation.shape != gradient.shape:
        raise ValueError(f"Activation shape {activation.shape} must match Gradient shape {gradient.shape}")
        
    n_slices, n_channels, height, width = activation.shape
    
    # 1. Global Average Pooling of Gradients (alpha weights)
    # (10, 960, 4, 4) -> (10, 960)
    # Use abs() or clamp(min=0) on gradients? Standard GradCAM uses raw gradients.
    alpha = torch.mean(gradient, dim=(2, 3)) 
    
    # 2. Weighted Combination of Activations
    # Need to reshape alpha for broadcasting: (10, 960) -> (10, 960, 1, 1)
    alpha = alpha.view(n_slices, n_channels, 1, 1)
    
    # Element-wise multiply alpha weights with activation maps
    # (10, 960, 1, 1) * (10, 960, 4, 4) -> (10, 960, 4, 4)
    weighted_activations = alpha * activation
    
    # 3. Sum across channels
    # (10, 960, 4, 4) -> (10, 4, 4)
    cam = torch.sum(weighted_activations, dim=1)
    
    # 4. Apply ReLU (important for Grad-CAM - only positive contributions)
    cam = F.relu(cam)
    
    # 5. Normalize heatmap per slice 0-1 (Crucial for visualization consistency)
    cam_flat = cam.view(n_slices, -1)
    cam_min = torch.min(cam_flat, dim=1, keepdim=True)[0]
    cam_max = torch.max(cam_flat, dim=1, keepdim=True)[0]
    
    # Reshape min/max for broadcasting: [10, 1] -> [10, 1, 1]
    cam_min = cam_min.view(n_slices, 1, 1)
    cam_max = cam_max.view(n_slices, 1, 1)

    # Normalize, adding epsilon to avoid division by zero if max == min
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8) 
    
    return cam # Shape [10, h, w]


def get_orientation_info(affine: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Determines axis orientations and finds the sagittal dimension index.
    Returns: 
        Tuple (sagittal_dim_index, y_dim_index, z_dim_index) or None if sagittal not found
    """
    # ... (function content unchanged) ...
    try:
        orientations = nib.aff2axcodes(affine) 
    except Exception as e: 
        print(f"Error getting orientation: {e}"); return None
    sagittal_dim, y_dim, z_dim = -1, -1, -1
    for i, code in enumerate(orientations):
        if code in ('L', 'R'): sagittal_dim = i
        elif code in ('A', 'P'): y_dim = i
        elif code in ('S', 'I'): z_dim = i
    if sagittal_dim == -1: print("Sagittal dim not found"); return None
    remaining_dims = [d for d in [0, 1, 2] if d != sagittal_dim]
    if len(remaining_dims) != 2: print("Unexpected remaining dims"); return None
    if y_dim == -1 and z_dim == -1: y_dim, z_dim = remaining_dims[0], remaining_dims[1]; print("Warning: Y/Z assigned heuristically.")
    elif y_dim == -1: y_dim = remaining_dims[0] if remaining_dims[0] != z_dim else remaining_dims[1]
    elif z_dim == -1: z_dim = remaining_dims[0] if remaining_dims[0] != y_dim else remaining_dims[1]
    if y_dim == z_dim or y_dim == -1 or z_dim == -1: print("Error assigning Y/Z dims"); return None
    print(f"Detected Orientations: {orientations} -> Sagittal(X):{sagittal_dim}, Y:{y_dim}, Z:{z_dim}")
    return sagittal_dim, y_dim, z_dim


def activation_and_gradient_to_nifti( # Renamed function
    data_path: str, # Now expects a dict {'activation': ..., 'gradient': ...}
    reference_nii_path: str,
    output_path: str,
):
    """
    [GRAD-CAM VERSION]
    Calculates Grad-CAM, converts 2D slice heatmaps into a 3D NIfTI,
    re-projecting them back into the space of the *original* T1 NIfTI.
    
    Args:
        data_path (str): Path to saved .pt file containing a dictionary 
                         with keys 'activation' and 'gradient'.
                         Shapes: [N_slices, C, h, w] (e.g., [10, 960, 4, 4])
        reference_nii_path (str): Path to the *original* T1 NIfTI file
        output_path (str): Path for the final output 3D NIfTI file
    """
    
    # --- Step 1: Load Activation and Gradient ---
    try:
        # Load the dictionary from CPU
        data_dict = torch.load(data_path, map_location=torch.device('cpu')) 
        activation = data_dict['activation']
        gradient = data_dict['gradient']
        print(f"Loaded activation (shape: {activation.shape}) and gradient (shape: {gradient.shape})")
    except FileNotFoundError: # ... (error handling unchanged) ...
        print(f"Error: Data file not found at {data_path}")
        return False 
    except KeyError: # ... (error handling unchanged) ...
        print(f"Error: Data file {data_path} must contain keys 'activation' and 'gradient'")
        return False
    except Exception as e: # ... (error handling unchanged) ...
        print(f"Error loading data file {data_path}: {e}")
        return False
        
    # Validate shapes
    if activation.ndim != 4 or activation.shape[0] != NUM_SLICES_PER_SUBJECT or \
       gradient.ndim != 4 or gradient.shape[0] != NUM_SLICES_PER_SUBJECT or \
       activation.shape[1:] != gradient.shape[1:]: # ... (validation unchanged) ...
        print(f"Error: Activation shape {activation.shape} or Gradient shape {gradient.shape} invalid.")
        return False
        
    # --- Step 2: Calculate Grad-CAM Heatmap ---
    print(f"Calculating Grad-CAM for {NUM_SLICES_PER_SUBJECT} slices...")
    with torch.no_grad(): # Ensure no gradients are tracked during heatmap calculation
      heatmap_small = calculate_gradcam(activation, gradient) # Shape [10, h, w]
      print(f"Calculated small heatmap shape: {heatmap_small.shape}")
    
    # --- Step 3: Load Reference & Get Orientation/Slicing Info ---
    try: # ... (logic unchanged) ...
        ref_img = nib.load(reference_nii_path)
        affine = ref_img.affine
        # Handle both 3D and 4D data - take only spatial dimensions
        if len(ref_img.shape) == 4:
            ref_3d_shape = ref_img.shape[:3]  # Take only spatial dimensions (X, Y, Z)
            print(f"4D reference data detected, using spatial shape: {ref_3d_shape}")
        else:
            ref_3d_shape = ref_img.shape
            print(f"3D reference data detected, shape: {ref_3d_shape}") 
        orientation_info = get_orientation_info(affine)
        if orientation_info is None: return False 
        sagittal_dim, y_dim, z_dim = orientation_info
        num_total_slices = ref_3d_shape[sagittal_dim]
        if num_total_slices < NUM_SLICES_PER_SUBJECT: return False # Error printed
        center_slice_index = num_total_slices // 2
        start_index = center_slice_index - (NUM_SLICES_PER_SUBJECT // 2)
        original_2d_shape = (ref_3d_shape[y_dim], ref_3d_shape[z_dim])
        print(f"Original 2D slice shape (Y={y_dim}, Z={z_dim}): {original_2d_shape}")
    except FileNotFoundError: print(f"Ref NIfTI not found: {reference_nii_path}"); return False 
    except Exception as e: print(f"Error reading ref NIfTI: {e}"); return False

    # --- Step 4: Upsample Heatmaps to ORIGINAL 2D Slice Size (Linear Interpolation) ---
    heatmap_small_np = heatmap_small.numpy() # ... (logic unchanged) ...
    resized_heatmaps = []
    target_cv2_size = (original_2d_shape[1], original_2d_shape[0]) 
    print(f"Upsampling {NUM_SLICES_PER_SUBJECT} heatmaps from {heatmap_small_np.shape[1:]} to {target_cv2_size}...")
    for i in range(NUM_SLICES_PER_SUBJECT):
        resized = cv2.resize(heatmap_small_np[i], target_cv2_size, 
                             interpolation=cv2.INTER_LINEAR) 
        resized_heatmaps.append(resized)
    resized_heatmaps_np = np.stack(resized_heatmaps)
    print(f"Upsampled heatmaps shape: {resized_heatmaps_np.shape}")

    # --- Step 5: Re-project 2D Slices into 3D Volume (Dynamic Axis, No Rotation Undo) ---
    output_volume = np.zeros(ref_3d_shape, dtype=np.float32) # ... (logic unchanged) ...
    print(f"Re-projecting {NUM_SLICES_PER_SUBJECT} slices into 3D volume (shape {ref_3d_shape}) starting at index {start_index} along dimension {sagittal_dim}...")
    successful_slices = 0
    for i in range(NUM_SLICES_PER_SUBJECT):
        slice_index = start_index + i
        heatmap_2d_to_paste = resized_heatmaps_np[i] 
        paste_indices: List[slice] = [slice(None)] * 3 
        paste_indices[sagittal_dim] = slice_index 
        target_slice_shape = tuple(s for idx, s in enumerate(ref_3d_shape) if idx != sagittal_dim)
        if heatmap_2d_to_paste.shape == target_slice_shape:
             try: output_volume[tuple(paste_indices)] = heatmap_2d_to_paste; successful_slices += 1
             except IndexError: print(f"Error: Indexing failed for slice {i}...") # Simplified
        else: print(f"Warning: Shape mismatch for slice {i}...") # Simplified
    if successful_slices == 0: print("Error: Failed to re-project any slices."); return False 
    elif successful_slices < NUM_SLICES_PER_SUBJECT: print(f"Warning: Only {successful_slices}/{NUM_SLICES_PER_SUBJECT} slices re-projected.")

    # --- Step 6: Save NIfTI ---
    # Normalization happened within calculate_gradcam
    try: # ... (logic unchanged) ...
        output_dir = os.path.dirname(output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        nifti_img = nib.Nifti1Image(output_volume.astype(np.float32), affine)
        nib.save(nifti_img, output_path)
        print(f"Saved 3D Grad-CAM heatmap to: {output_path}")
        return True 
    except Exception as e: print(f"Error saving NIfTI: {e}"); return False 


if __name__ == "__main__":
    
    # --- Configuration for the Test ---
    # Path to where activation AND gradient are saved (as a dictionary)
    DATA_PATH = "output/papermodel_test/act_and_grad_backbone.stage4.1.gconv2.pt" 
    
    # Path to the ORIGINAL T1 NIfTI file used as input to the model
    REF_PATH = "/Volumes/3T-disk/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii" 
    
    # Path for the final 3D Grad-CAM heatmap NIfTI output
    OUT_PATH = "output/papermodel_test/gradcam_heatmap_3D.nii.gz" 

    # --- Mock data generation (Save dict with activation AND gradient) ---
    if not os.path.exists(os.path.dirname(DATA_PATH)): os.makedirs(os.path.dirname(DATA_PATH))
    if not os.path.exists(DATA_PATH):
        print(f"Mocking activation and gradient file at: {DATA_PATH}")
        # Example shape for stage4 output: [10, 960, 4, 4]
        mock_act = torch.randn(NUM_SLICES_PER_SUBJECT, 960, 4, 4) 
        # Make mock gradients positive to ensure ReLU in GradCAM doesn't zero everything out
        mock_grad = torch.abs(torch.randn(NUM_SLICES_PER_SUBJECT, 960, 4, 4)) 
        mock_data = {'activation': mock_act, 'gradient': mock_grad}
        torch.save(mock_data, DATA_PATH)
    # --- End Mock ---

    # --- Run the main function ---
    if not os.path.exists(REF_PATH):
        print(f"Error: Reference NIfTI file not found at {REF_PATH}")
    else:
        print(f"--- Running Grad-CAM Heatmap Generation ---")
        success = activation_and_gradient_to_nifti(
            data_path=DATA_PATH,
            reference_nii_path=REF_PATH,
            output_path=OUT_PATH,
        )
        if success:
             print("--- Script finished successfully ---")
        else:
             print("--- Script finished with errors ---")