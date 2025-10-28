import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import os
import cv2 # Needed for resizing
from typing import List, Dict, Any, Tuple, Optional # Added for type hinting clarity

# (Import constants from model script)
try:
    # We need these constants to correctly determine the original slice shape
    # before the final resize.
    from model.shufflenet.model import NUM_SLICES_PER_SUBJECT, SLICE_IMG_SIZE 
except ImportError as e:
    print(f"Error: Could not import constants from model.shufflenet.model")
    print(f"Details: {e}")
    # Provide default values so the script can still be loaded, but warn user
    NUM_SLICES_PER_SUBJECT = 10
    SLICE_IMG_SIZE = 128
    print("Warning: Using default values for NUM_SLICES_PER_SUBJECT=10, SLICE_IMG_SIZE=128.")


def get_orientation_info(affine: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Determines axis orientations and finds the sagittal dimension index.
    Returns: 
        Tuple (sagittal_dim_index, y_dim_index, z_dim_index) or None if sagittal not found
    """
    try:
        # Get axis codes ('R', 'A', 'S', 'L', 'P', 'I')
        orientations = nib.aff2axcodes(affine) 
    except Exception as e:
        print(f"Error getting orientation from affine: {e}")
        return None
        
    sagittal_dim = -1
    y_dim = -1
    z_dim = -1
    
    # Find which numpy dimension (0, 1, or 2) corresponds to Left-Right
    for i, code in enumerate(orientations):
        if code in ('L', 'R'):
            sagittal_dim = i
        # Heuristic assignment for Y and Z based on common neuroimaging conventions
        # This might need adjustment if your data uses a very unusual orientation
        elif code in ('A', 'P'): # Anterior-Posterior often corresponds to Y in sagittal view
            y_dim = i
        elif code in ('S', 'I'): # Superior-Inferior often corresponds to Z in sagittal view
            z_dim = i
            
    if sagittal_dim == -1:
        print(f"Error: Could not determine sagittal dimension from orientations: {orientations}")
        return None
        
    # Determine which of the remaining dimensions corresponds to Y and Z if not found directly
    remaining_dims = [d for d in [0, 1, 2] if d != sagittal_dim]
    if len(remaining_dims) != 2:
         print(f"Error: Unexpected number of remaining dimensions ({len(remaining_dims)}) after finding sagittal dim {sagittal_dim}.")
         return None
         
    if y_dim == -1 and z_dim == -1:
        y_dim, z_dim = remaining_dims[0], remaining_dims[1] # Assign remaining arbitrarily
        print(f"Warning: Could not definitively identify Y/Z axes from orientation codes {orientations}. Assigning remaining dims {remaining_dims} heuristically.")
    elif y_dim == -1:
         y_dim = remaining_dims[0] if remaining_dims[0] != z_dim else remaining_dims[1]
    elif z_dim == -1:
         z_dim = remaining_dims[0] if remaining_dims[0] != y_dim else remaining_dims[1]
         
    if y_dim == z_dim: # Check if somehow y and z got assigned the same index
        print(f"Error: Y dimension ({y_dim}) and Z dimension ({z_dim}) were assigned the same index.")
        return None

    print(f"Detected Orientations: {orientations}")
    print(f"-> Sagittal dimension index (X): {sagittal_dim}")
    print(f"-> Assumed Y dimension index: {y_dim}")
    print(f"-> Assumed Z dimension index: {z_dim}")
    
    return sagittal_dim, y_dim, z_dim


def activation_to_nifti( # Renamed function
    activation_path: str,
    reference_nii_path: str,
    output_path: str,
):
    """
    [FINAL VERSION]
    Converts 2D slice activations (from PaperModel's backbone) into a 3D NIfTI heatmap,
    re-projecting them back into the space of the *original* T1 NIfTI.
    
    Uses 'mean channel aggregation' (can be replaced with Grad-CAM later).
    Uses Linear interpolation for resizing heatmaps.
    Does NOT attempt to undo the np.rot90 from preprocessing.
    Dynamically detects orientation.
    
    Args:
        activation_path (str): Path to saved .pt tensor, shape [10, C, h, w] (e.g., [10, 960, 4, 4])
        reference_nii_path (str): Path to the *original* T1 NIfTI file (e.g., T1_3D_..._3b.nii)
        output_path (str): Path for the final output 3D NIfTI file
    """
    
    # --- Step 1: Load Activation ---
    try:
        # Load from CPU, as saved by the hook
        act = torch.load(activation_path, map_location=torch.device('cpu')) 
    except FileNotFoundError:
        print(f"Error: Activation file not found at {activation_path}")
        return False # Indicate failure
    except Exception as e:
        print(f"Error loading activation file {activation_path}: {e}")
        return False
        
    if act.ndim != 4 or act.shape[0] != NUM_SLICES_PER_SUBJECT:
        print(f"Error: Activation shape {act.shape}. Expected [{NUM_SLICES_PER_SUBJECT}, C, h, w]")
        return False
        
    # --- Step 2: Aggregate Channels (Simple Mean) ---
    # (10, C, h, w) -> (10, h, w) e.g., [10, 4, 4]
    with torch.no_grad(): # Ensure no gradients are tracked
      heatmap_small = torch.mean(act, dim=1) 
    
    # --- Step 3: Load Reference & Get Orientation/Slicing Info ---
    try:
        ref_img = nib.load(reference_nii_path)
        affine = ref_img.affine
        ref_3d_shape = ref_img.shape # (Dim0, Dim1, Dim2) e.g. (192, 256, 256)
        
        orientation_info = get_orientation_info(affine)
        if orientation_info is None:
            return False # Error already printed
        sagittal_dim, y_dim, z_dim = orientation_info
        
        # Calculate start index based on the *detected* sagittal dimension
        num_total_slices = ref_3d_shape[sagittal_dim]
        if num_total_slices < NUM_SLICES_PER_SUBJECT:
            print(f"Error: Ref NIfTI has only {num_total_slices} slices in sagittal dim ({sagittal_dim}), less than {NUM_SLICES_PER_SUBJECT}.")
            return False
            
        center_slice_index = num_total_slices // 2
        start_index = center_slice_index - (NUM_SLICES_PER_SUBJECT // 2)
        
        # Get the original 2D slice shape using the detected Y and Z dimensions' sizes
        original_2d_shape = (ref_3d_shape[y_dim], ref_3d_shape[z_dim])
        print(f"Original 2D slice shape (Y={y_dim}, Z={z_dim}): {original_2d_shape}")
        
    except FileNotFoundError:
        print(f"Error: Reference NIfTI not found at {reference_nii_path}")
        return False
    except Exception as e:
        print(f"Error reading reference NIfTI or determining orientation: {e}")
        return False

    # --- Step 4: Upsample Heatmaps to ORIGINAL 2D Slice Size (Linear Interpolation) ---
    heatmap_small_np = heatmap_small.numpy()
    resized_heatmaps = []
    # cv2 wants (width, height) i.e. (Z_dim_size, Y_dim_size)
    target_cv2_size = (original_2d_shape[1], original_2d_shape[0]) 
    
    print(f"Upsampling {NUM_SLICES_PER_SUBJECT} heatmaps from {heatmap_small_np.shape[1:]} to {target_cv2_size}...")
    for i in range(NUM_SLICES_PER_SUBJECT):
        # Use INTER_LINEAR
        resized = cv2.resize(heatmap_small_np[i], target_cv2_size, 
                             interpolation=cv2.INTER_LINEAR) 
        resized_heatmaps.append(resized)
        
    # Stack back into a numpy array (10, Y_size, Z_size) e.g., [10, 256, 256]
    resized_heatmaps_np = np.stack(resized_heatmaps)

    # --- Step 5: Re-project 2D Slices into 3D Volume (Dynamic Axis, No Rotation Undo) ---
    output_volume = np.zeros(ref_3d_shape, dtype=np.float32)

    print(f"Re-projecting {NUM_SLICES_PER_SUBJECT} slices into 3D volume (shape {ref_3d_shape}) starting at index {start_index} along dimension {sagittal_dim}...")
    successful_slices = 0
    for i in range(NUM_SLICES_PER_SUBJECT):
        slice_index = start_index + i
        
        # Get the resized heatmap directly (NO np.rot90 undo)
        heatmap_2d_to_paste = resized_heatmaps_np[i] 

        # Create slice indices for pasting: [:, :, :]
        paste_indices: List[slice] = [slice(None)] * 3 
        paste_indices[sagittal_dim] = slice_index # Set the correct slice index
        
        # Check if the shape matches the target slice shape in the volume
        target_slice_shape = tuple(s for idx, s in enumerate(ref_3d_shape) if idx != sagittal_dim)
        if heatmap_2d_to_paste.shape == target_slice_shape:
             try:
                output_volume[tuple(paste_indices)] = heatmap_2d_to_paste 
                successful_slices += 1
             except IndexError:
                 print(f"Error: Indexing failed for slice {i} at index {slice_index} along dim {sagittal_dim}. Check dimensions.")
        else:
             print(f"Warning: Shape mismatch during pasting slice {i}. Heatmap shape {heatmap_2d_to_paste.shape}, Target slice shape {target_slice_shape}. Skipping slice.")

    if successful_slices == 0:
        print("Error: Failed to re-project any slices. Aborting save.")
        return False
    elif successful_slices < NUM_SLICES_PER_SUBJECT:
         print(f"Warning: Only {successful_slices}/{NUM_SLICES_PER_SUBJECT} slices were successfully re-projected.")

    # --- Step 6: Normalize & Save NIfTI ---
    # Normalize the final volume from 0 to 1 for better visualization
    v_min = output_volume.min()
    v_max = output_volume.max()
    if v_max > v_min + 1e-8: # Avoid division by zero or near-zero
        output_volume = (output_volume - v_min) / (v_max - v_min)
    else:
        print("Warning: Output volume is nearly constant; skipping normalization.")
        # Output volume remains as is (likely all zeros or a constant value)
    
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: # Ensure output directory is not empty
             os.makedirs(output_dir, exist_ok=True)
             
        # Create NIfTI image object using the output volume and original affine
        nifti_img = nib.Nifti1Image(output_volume.astype(np.float32), affine)
        nib.save(nifti_img, output_path)
        print(f"Saved 3D re-projected heatmap to: {output_path}")
        return True # Indicate success
        
    except Exception as e:
        print(f"Error saving NIfTI file to {output_path}: {e}")
        return False


if __name__ == "__main__":
    
    # --- Configuration for the Test ---
    # Path to a saved activation file from PaperModel's backbone
    ACT_PATH = "output/papermodel_test/backbone.stage4.1.gconv2.pt" 
    
    # Path to the ORIGINAL T1 NIfTI file used as input to the model
    REF_PATH = "/Volumes/3T-disk/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii" 
    
    # Path for the final 3D heatmap NIfTI output
    OUT_PATH = "output/papermodel_test/heatmap_3D_final.nii.gz" # New output name

    # --- Mock activation file generation (if needed for standalone test) ---
    if not os.path.exists(os.path.dirname(ACT_PATH)): os.makedirs(os.path.dirname(ACT_PATH))
    if not os.path.exists(ACT_PATH):
        print(f"Mocking activation file at: {ACT_PATH}")
        # Shape [10, C, h, w], e.g., [10, 960, 4, 4] for stage4 output
        mock_act = torch.randn(NUM_SLICES_PER_SUBJECT, 960, 4, 4) 
        torch.save(mock_act, ACT_PATH)
    # --- End Mock ---

    # --- Run the main function ---
    if not os.path.exists(REF_PATH):
        print(f"Error: Reference NIfTI file not found at {REF_PATH}")
        print("Please update REF_PATH to the actual original T1 NIfTI file.")
    else:
        print(f"--- Running Activation to NIfTI Reprojection (Final Version) ---")
        success = activation_to_nifti(
            activation_path=ACT_PATH,
            reference_nii_path=REF_PATH,
            output_path=OUT_PATH,
        )
        if success:
             print("--- Script finished successfully ---")
        else:
             print("--- Script finished with errors ---")