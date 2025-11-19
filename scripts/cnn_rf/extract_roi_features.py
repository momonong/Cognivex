"""
ROI Feature Extraction from MRI Images

Extract ROI features from raw MRI images (GM, FA, MD) using AAL3 atlas.
This is the first step in the CNN-RF pipeline.

Input: MRI images in data/MRI_processed/
Output: ROI features CSV file
"""

import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import json
from tqdm import tqdm
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ROIFeatureExtractor:
    """Extract ROI features from MRI images using AAL3 atlas"""
    
    def __init__(
        self,
        atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
        atlas_labels_path="data/aal3/AAL3v1.json"
    ):
        """
        Initialize feature extractor
        
        Args:
            atlas_path: Path to AAL3 atlas NIfTI file
            atlas_labels_path: Path to AAL3 labels JSON file
        """
        self.atlas_path = Path(atlas_path)
        self.atlas_labels_path = Path(atlas_labels_path)
        
        # Load atlas
        print(f"[INFO] Loading AAL3 atlas from: {self.atlas_path}")
        self.atlas_img = nib.load(self.atlas_path)
        self.atlas_data = self.atlas_img.get_fdata().astype(int)
        print(f"[OK] Atlas loaded: shape={self.atlas_data.shape}")
        
        # Load labels
        print(f"[INFO] Loading AAL3 labels from: {self.atlas_labels_path}")
        with open(self.atlas_labels_path, 'r', encoding='utf-8') as f:
            labels_raw = json.load(f)
        
        # Create index -> name mapping
        self.roi_names = {int(idx): name for idx, name in labels_raw.items()}
        self.roi_indices = sorted(self.roi_names.keys())
        print(f"[OK] Loaded {len(self.roi_names)} ROI labels")
        
        # Cache for resampled atlas
        self.atlas_cache = {}
    
    def get_resampled_atlas(self, target_shape):
        """
        Get atlas resampled to target shape (with caching)
        
        Args:
            target_shape: Target shape tuple
        
        Returns:
            Resampled atlas data
        """
        # Check cache
        cache_key = tuple(target_shape)
        if cache_key in self.atlas_cache:
            return self.atlas_cache[cache_key]
        
        # Resample atlas
        from scipy.ndimage import zoom
        
        zoom_factors = [
            target_shape[i] / self.atlas_data.shape[i]
            for i in range(3)
        ]
        
        print(f"[INFO] Resampling atlas: {self.atlas_data.shape} -> {target_shape}")
        atlas_resampled = zoom(
            self.atlas_data.astype(float),
            zoom_factors,
            order=0  # Nearest neighbor to preserve labels
        ).astype(int)
        
        # Cache result
        self.atlas_cache[cache_key] = atlas_resampled
        print(f"[OK] Atlas resampled and cached")
        
        return atlas_resampled
    
    def extract_roi_mean(self, mri_img, roi_index, atlas_resampled=None):
        """
        Extract mean value from a specific ROI
        
        Args:
            mri_img: NIfTI image
            roi_index: ROI index in atlas
            atlas_resampled: Pre-resampled atlas (optional, for speed)
        
        Returns:
            Mean value in the ROI
        """
        mri_data = mri_img.get_fdata()
        
        # Use provided resampled atlas or get from cache
        if atlas_resampled is None:
            if mri_data.shape != self.atlas_data.shape:
                atlas_resampled = self.get_resampled_atlas(mri_data.shape)
            else:
                atlas_resampled = self.atlas_data
        
        # Get mask for this ROI
        roi_mask = atlas_resampled == roi_index
        
        # Extract values
        roi_values = mri_data[roi_mask]
        
        if len(roi_values) == 0:
            return 0.0
        
        return float(np.mean(roi_values))
    
    def extract_subject_features(self, subject_dir):
        """
        Extract features for one subject
        
        Args:
            subject_dir: Path to subject directory containing MRI files
        
        Returns:
            Dictionary of features {roi_name_modality: value}
        """
        subject_dir = Path(subject_dir)
        subject_id = subject_dir.name
        
        # Find MRI files
        gm_file = list(subject_dir.glob("*_GM_to_MNI.nii.gz"))
        fa_file = list(subject_dir.glob("*_FA_to_MNI.nii.gz"))
        md_file = list(subject_dir.glob("*_MD_to_MNI.nii.gz"))
        
        if not (gm_file and fa_file and md_file):
            raise FileNotFoundError(
                f"Missing MRI files for {subject_id}. "
                f"Found: GM={len(gm_file)}, FA={len(fa_file)}, MD={len(md_file)}"
            )
        
        # Load MRI images
        gm_img = nib.load(gm_file[0])
        fa_img = nib.load(fa_file[0])
        md_img = nib.load(md_file[0])
        
        # Get resampled atlas once (for all modalities)
        mri_shape = gm_img.get_fdata().shape
        if mri_shape != self.atlas_data.shape:
            atlas_resampled = self.get_resampled_atlas(mri_shape)
        else:
            atlas_resampled = self.atlas_data
        
        # Extract features for each ROI
        features = {}
        
        for roi_idx in self.roi_indices:
            roi_name = self.roi_names[roi_idx]
            
            # Extract mean values for each modality (reuse resampled atlas)
            gm_mean = self.extract_roi_mean(gm_img, roi_idx, atlas_resampled)
            fa_mean = self.extract_roi_mean(fa_img, roi_idx, atlas_resampled)
            md_mean = self.extract_roi_mean(md_img, roi_idx, atlas_resampled)
            
            # Store features
            features[f"{roi_name}_GM"] = gm_mean
            features[f"{roi_name}_FA"] = fa_mean
            features[f"{roi_name}_MD"] = md_mean
        
        return features
    
    def extract_dataset_features(
        self,
        data_root="data/MRI_processed",
        output_csv="data/roi_features.csv"
    ):
        """
        Extract features for entire dataset
        
        Args:
            data_root: Root directory containing NC/MCI/AD folders
            output_csv: Output CSV file path
        
        Returns:
            DataFrame with all features
        """
        data_root = Path(data_root)
        
        print(f"\n[INFO] Extracting features from: {data_root}")
        
        # Collect all subjects
        all_features = []
        
        for group in ['NC', 'MCI', 'AD']:
            group_dir = data_root / group
            
            if not group_dir.exists():
                print(f"[WARN] Group directory not found: {group_dir}")
                continue
            
            # Get all subject directories
            subject_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir()])
            
            print(f"\n[INFO] Processing {group}: {len(subject_dirs)} subjects")
            
            for subject_dir in tqdm(subject_dirs, desc=f"Extracting {group}"):
                try:
                    # Extract features
                    features = self.extract_subject_features(subject_dir)
                    
                    # Add metadata
                    features['Subject_ID'] = subject_dir.name
                    features['Group'] = group
                    
                    all_features.append(features)
                    
                except Exception as e:
                    print(f"\n[ERROR] Failed to process {subject_dir.name}: {e}")
                    continue
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Reorder columns: Subject_ID, Group, then features
        feature_cols = [col for col in df.columns if col not in ['Subject_ID', 'Group']]
        df = df[['Subject_ID', 'Group'] + sorted(feature_cols)]
        
        # Save to CSV
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        print(f"\n[OK] Features extracted for {len(df)} subjects")
        print(f"[OK] Saved to: {output_csv}")
        print(f"\nDataset summary:")
        print(df['Group'].value_counts().to_string())
        print(f"\nFeature dimensions: {len(feature_cols)} features")
        
        return df


def main():
    """Main feature extraction function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ROI features from MRI images")
    parser.add_argument(
        '--data-root',
        default='data/MRI_processed',
        help='Root directory containing NC/MCI/AD folders'
    )
    parser.add_argument(
        '--output',
        default='data/roi_features.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--atlas',
        default='data/aal3/AAL3v1_1mm.nii.gz',
        help='Path to AAL3 atlas'
    )
    parser.add_argument(
        '--labels',
        default='data/aal3/AAL3v1.json',
        help='Path to AAL3 labels JSON'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("ROI Feature Extraction")
    print("="*80)
    
    # Create extractor
    extractor = ROIFeatureExtractor(
        atlas_path=args.atlas,
        atlas_labels_path=args.labels
    )
    
    # Extract features
    df = extractor.extract_dataset_features(
        data_root=args.data_root,
        output_csv=args.output
    )
    
    print("\n" + "="*80)
    print("[SUCCESS] Feature extraction completed!")
    print("="*80)


if __name__ == "__main__":
    main()
