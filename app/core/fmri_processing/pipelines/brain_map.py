from collections import defaultdict
import numpy as np
import nibabel as nib
import pandas as pd


def analyze_brain_activation(
    activation_path: str,
    atlas_path: str,
    label_path: str,
) -> pd.DataFrame:
    """
    Analyze activation overlaps with brain atlas regions.
    Returns a DataFrame summarizing the activation statistics.
    """
    # Load atlas image
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)

    # Load label mapping
    id_to_label = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label_id = int(parts[0])
                label_name = " ".join(parts[1:])
                id_to_label[label_id] = label_name

    # Load activation image
    act_img = nib.load(activation_path)
    act_data = act_img.get_fdata()

    # Get activated voxels
    activated_voxels = np.where(act_data > 0)
    act_values = act_data[activated_voxels]
    atlas_labels = atlas_data[activated_voxels]

    # Accumulate activation and voxel count per region
    label_activation_sum = defaultdict(float)
    label_voxel_count = defaultdict(int)

    for label_id, act_value in zip(atlas_labels, act_values):
        label_id = int(label_id)
        if label_id == 0:
            continue  # Skip background
        label_activation_sum[label_id] += act_value
        label_voxel_count[label_id] += 1

    # Construct result table
    results = []
    for label_id in sorted(
        label_activation_sum.keys(), key=lambda k: label_activation_sum[k], reverse=True
    ):
        name = id_to_label.get(label_id, "Unknown")
        total = label_activation_sum[label_id]
        count = label_voxel_count[label_id]
        avg = total / count if count > 0 else 0
        results.append(
            {
                "Label ID": label_id,
                "Region Name": name,
                "Voxel Count": count,
                "Total Activation": total,
                "Mean Activation": avg,
            }
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Mean Activation", ascending=False).reset_index(drop=True)

    return df


if __name__ == "__main__":
    import os

    # --- UPDATED Paths for ROI Analysis ---

    # Input: The final heatmap, resampled and aligned to the atlas grid
    #        (Output from the previous 'resample.py' step)
    FINAL_HEATMAP_PATH = "output/single_subject_final_resampled_accurate/subject_008_ants_heatmap_MNI_accurate_resampled_to_AAL3v1_1mm.nii.gz"

    # Atlas: The NIfTI file defining the brain regions
    ATLAS_NIFTI_PATH = "data/aal3/AAL3v1_1mm.nii.gz"

    # Labels: The text file mapping atlas integer labels to region names
    ATLAS_LABEL_PATH = "data/aal3/AAL3v1_1mm.nii.txt"  # Make sure this file exists!

    # --- Check files ---
    if not os.path.exists(FINAL_HEATMAP_PATH):
        print(f"Error: Final heatmap NIfTI not found at {FINAL_HEATMAP_PATH}")
    elif not os.path.exists(ATLAS_NIFTI_PATH):
        print(f"Error: Atlas NIfTI not found at {ATLAS_NIFTI_PATH}")
    elif not os.path.exists(ATLAS_LABEL_PATH):
        print(f"Error: Atlas label file not found at {ATLAS_LABEL_PATH}")
    else:
        print("--- Running Brain Activation Analysis ---")
        df_result = analyze_brain_activation(
            activation_path=FINAL_HEATMAP_PATH,
            atlas_path=ATLAS_NIFTI_PATH,
            label_path=ATLAS_LABEL_PATH,
        )

        print("\n--- Analysis Results (Top 10 Regions by Total Activation) ---")
        # Display the top results
        print(df_result.head(10).to_string())

        # (Optional) Save the full results to a CSV file
        output_csv_path = (
            "output/single_subject_final_resampled/roi_activation_analysis.csv"
        )
        df_result.to_csv(output_csv_path, index=False)
        print(f"\nFull results saved to: {output_csv_path}")

    print(df_result)
