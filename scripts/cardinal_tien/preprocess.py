import os
import glob
import ants # Import ANTsPy
import numpy as np # Needed for find_t1_files helper

# --- 1. Configuration (MODIFY THESE PATHS!) ---

# Base directory containing AD/ and NC/ folders
BASE_DATA_DIR = "/Volumes/3T-disk/fMRI/Model/sMRI_data" 

# Directory where the native-space heatmaps from act_to_nii_final.py are saved
# ASSUMPTION: Saved like <NATIVE_HEATMAP_DIR>/<SubjectID>/heatmap_3D_final.nii.gz
NATIVE_HEATMAP_DIR = "output/papermodel_test" 

# Directory to save the final normalized (MNI-space) heatmaps
OUTPUT_NORMALIZED_DIR = "output/normalized_heatmaps_ants"

# Path to your downloaded MNI T1 template NIfTI file (e.g., MNI152_T1_1mm.nii.gz)
MNI_TEMPLATE_PATH = "data/aal3/AAL3v1_1mm.nii.gz" 

# ANTs Registration Type (e.g., 'SyN' for accuracy, 'Affine'/'Rigid' for speed)
# 'SyN' (Symmetric Normalization) is recommended for T1 normalization.
ANTS_TRANSFORM_TYPE = 'SyN' 

# Interpolator for applying transform to heatmap ('linear' is good)
ANTS_INTERPOLATOR = 'linear'

# --- End Configuration ---

# --- Helper function to find files ---
def find_t1_files(base_dir):
    """ Finds all T1 NIfTI files within the AD/NC subdirectories """
    t1_files = []
    # Search for .nii and .nii.gz recursively within AD and NC folders
    for class_folder in ["AD", "NC"]:
        search_path = os.path.join(base_dir, class_folder, "*", "*.nii*") # Look inside subject folders
        found = glob.glob(search_path, recursive=False) # Don't go deeper than subject folder
        # Filter for files likely being the main T1 (simple heuristic)
        t1_files.extend([f for f in found if 'T1_3D_mprage_SAG' in os.path.basename(f) and not f.endswith('.json')])
    return t1_files

# --- Main Batch Processing Logic ---
if __name__ == "__main__":
    
    # --- Check MNI Template ---
    if not os.path.exists(MNI_TEMPLATE_PATH):
        print(f"Error: MNI Template file not found at {MNI_TEMPLATE_PATH}")
        print("Please download an MNI T1 template (e.g., MNI152_T1_1mm.nii.gz) and update MNI_TEMPLATE_PATH.")
        exit()

    # --- Load Fixed Image (MNI Template) ---
    try:
        print(f"Loading fixed MNI template: {MNI_TEMPLATE_PATH}")
        fixed_image = ants.image_read(MNI_TEMPLATE_PATH)
    except Exception as e:
        print(f"Error loading MNI template: {e}")
        exit()

    # --- Find all T1 files ---
    print(f"Searching for T1 files in: {BASE_DATA_DIR}")
    t1_files = find_t1_files(BASE_DATA_DIR)
    if not t1_files:
        print("Error: No T1 files found. Check BASE_DATA_DIR.")
        exit()
    print(f"Found {len(t1_files)} T1 files.")

    # --- Create main output directory ---
    os.makedirs(OUTPUT_NORMALIZED_DIR, exist_ok=True)
    
    # --- Loop through each T1 file ---
    processed_count = 0
    error_count = 0
    for t1_path in t1_files:
        subject_dir_name = os.path.basename(os.path.dirname(t1_path)) # e.g., T1_3D_MPRAGE_SAG_0003_008
        print(f"\n--- Processing: {subject_dir_name} ---")
        
        # --- Construct expected heatmap path ---
        heatmap_path = os.path.join(NATIVE_HEATMAP_DIR, subject_dir_name, "heatmap_3D_final.nii.gz") # Adjust filename if needed
        
        # Define final output path for this subject
        final_output_path = os.path.join(OUTPUT_NORMALIZED_DIR, subject_dir_name, "heatmap_3D_MNI_ants.nii.gz")
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

        # Check if files exist
        if not os.path.exists(t1_path):
             print(f"  Error: T1 file missing: {t1_path}. Skipping.")
             error_count += 1
             continue
        if not os.path.exists(heatmap_path):
            print(f"  Warning: Native heatmap not found at {heatmap_path}. Skipping.")
            error_count += 1
            continue
            
        # --- Load Moving Images (T1 and Heatmap) ---
        try:
            print(f"  Loading moving T1: {os.path.basename(t1_path)}")
            moving_image = ants.image_read(t1_path)
            print(f"  Loading native heatmap: {os.path.basename(heatmap_path)}")
            heatmap_native = ants.image_read(heatmap_path)
        except Exception as e:
            print(f"  Error loading images for {subject_dir_name}: {e}. Skipping.")
            error_count += 1
            continue

        # --- Calculate Transformation ---
        try:
            print(f"  Calculating registration (Transform: {ANTS_TRANSFORM_TYPE})... (This may take time)")
            # Perform registration: moving (T1) -> fixed (MNI)
            transform = ants.registration(fixed=fixed_image, 
                                          moving=moving_image, 
                                          type_of_transform=ANTS_TRANSFORM_TYPE) 
            print("  Registration calculated.")
            # transform is a dictionary like {'fwdtransforms': [warp_path, affine_path], 'invtransforms': [...]}
            
        except Exception as e:
            print(f"  Error during ANTs registration for {subject_dir_name}: {e}. Skipping.")
            error_count += 1
            continue

        # --- Apply Transformation to Heatmap ---
        try:
            print(f"  Applying transform to heatmap (Interpolator: {ANTS_INTERPOLATOR})...")
            # Apply the forward transforms (native -> MNI) to the native heatmap
            heatmap_normalized = ants.apply_transforms(fixed=fixed_image, 
                                                       moving=heatmap_native,
                                                       transformlist=transform['fwdtransforms'],
                                                       interpolator=ANTS_INTERPOLATOR)
            print("  Transform applied.")
            
            # --- Save Normalized Heatmap ---
            print(f"  Saving normalized heatmap to: {final_output_path}")
            ants.image_write(heatmap_normalized, final_output_path)
            processed_count += 1

        except Exception as e:
             print(f"  Error applying ANTs transform or saving for {subject_dir_name}: {e}. Skipping.")
             error_count += 1
             
    # --- Final Summary ---
    print("\n--- Batch ANTs Normalization Complete ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors/Skipped: {error_count}")
    print(f"Normalized heatmaps saved in subdirectories under: {OUTPUT_NORMALIZED_DIR}")