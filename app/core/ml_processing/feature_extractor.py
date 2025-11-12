"""
ROI feature extraction for structural MRI analysis
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Optional
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker

from .exceptions import FeatureExtractionError, AtlasLoadError


class ROIFeatureExtractor:
    """
    Extracts ROI features from T1-weighted structural MRI
    
    Uses AAL (Automated Anatomical Labeling) atlas to define brain regions
    and extracts mean intensity values for each ROI.
    """
    
    def __init__(self, atlas_name: str = "AAL"):
        """
        Initialize feature extractor
        
        Args:
            atlas_name: Name of the atlas to use (default: "AAL")
        """
        self.atlas_name = atlas_name
        self._atlas_img = None
        self._atlas_labels = None
        self._masker = None
        self._roi_mapping = None
    
    def load_atlas(self):
        """
        Load AAL atlas
        
        Returns:
            Atlas image and labels
            
        Raises:
            AtlasLoadError: If atlas loading fails
        """
        if self._atlas_img is not None:
            return self._atlas_img, self._atlas_labels
        
        try:
            print(f"\n=== Loading {self.atlas_name} Atlas ===")
            
            # Load AAL atlas from nilearn
            atlas = datasets.fetch_atlas_aal(version='SPM12')
            
            self._atlas_img = atlas['maps']
            self._atlas_labels = atlas['labels']
            
            print(f"✓ Loaded {self.atlas_name} atlas")
            print(f"  - Total regions: {len(self._atlas_labels)}")
            print("=" * 35 + "\n")
            
            return self._atlas_img, self._atlas_labels
            
        except Exception as e:
            raise AtlasLoadError(
                f"Failed to load {self.atlas_name} atlas: {e}. "
                f"Please ensure nilearn is installed and can download the atlas."
            )
    
    def get_roi_mapping(self) -> Dict[str, int]:
        """
        Get mapping from ROI names to atlas indices
        
        Returns:
            Dictionary mapping ROI names to their indices in the atlas
        """
        if self._roi_mapping is not None:
            return self._roi_mapping
        
        # Ensure atlas is loaded
        if self._atlas_labels is None:
            self.load_atlas()
        
        # Create mapping
        self._roi_mapping = {
            label: idx + 1  # AAL indices start from 1
            for idx, label in enumerate(self._atlas_labels)
        }
        
        return self._roi_mapping
    
    def _validate_roi_list(self, roi_list: List[str]) -> None:
        """
        Validate that all ROIs in the list exist in the atlas
        
        Args:
            roi_list: List of ROI names to validate
            
        Raises:
            FeatureExtractionError: If any ROI is not found in atlas
        """
        roi_mapping = self.get_roi_mapping()
        invalid_rois = [roi for roi in roi_list if roi not in roi_mapping]
        
        if invalid_rois:
            raise FeatureExtractionError(
                f"Invalid ROI names not found in {self.atlas_name} atlas: {invalid_rois}. "
                f"Available ROIs: {list(roi_mapping.keys())[:10]}... (showing first 10)"
            )
    
    def _ensure_mni_space(self, img_path: str):
        """
        Ensure the input image is in MNI152 space
        
        Args:
            img_path: Path to input image
            
        Returns:
            Image in MNI152 space (resampled if necessary)
        """
        import nibabel as nib
        from nilearn import image as nimg
        
        # Load input image
        img = nib.load(img_path)
        
        # Check if already in MNI space (approximately)
        # MNI152 has shape around (91, 109, 91) or (182, 218, 182) for 2mm/1mm
        img_shape = img.shape[:3]
        
        # Common MNI shapes
        mni_shapes = [
            (91, 109, 91),    # 2mm MNI
            (182, 218, 182),  # 1mm MNI
            (193, 229, 193),  # SPM MNI
        ]
        
        # If already in MNI-like space, return as is
        if img_shape in mni_shapes:
            print(f"  ✓ Image already in MNI space: {img_shape}")
            return img
        
        # Otherwise, resample to MNI space
        print(f"  ⚠️  Image shape {img_shape} not standard MNI")
        print(f"  → Resampling to MNI152 space...")
        
        # Resample to match atlas
        if self._atlas_img is None:
            self.load_atlas()
        
        # Resample input image to atlas space
        img_resampled = nimg.resample_to_img(
            img, 
            self._atlas_img,
            interpolation='continuous'
        )
        
        print(f"  ✓ Resampled to: {img_resampled.shape}")
        return img_resampled
    
    def extract_features(
        self, 
        nii_path: str, 
        roi_list: List[str],
        standardize: bool = False,
        ensure_mni: bool = True
    ) -> np.ndarray:
        """
        Extract ROI features from T1 MRI image
        
        Args:
            nii_path: Path to T1 MRI NIfTI file
            roi_list: List of ROI names to extract
            standardize: Whether to standardize features (default: False, 
                        as we use external scaler)
            ensure_mni: Whether to ensure image is in MNI space (default: True)
        
        Returns:
            Feature vector of shape (n_rois,) containing mean intensity 
            for each ROI
            
        Raises:
            FeatureExtractionError: If extraction fails
        """
        try:
            print(f"\n=== Extracting ROI Features ===")
            print(f"Input: {nii_path}")
            print(f"ROIs to extract: {len(roi_list)}")
            
            # Validate input file
            nii_path = Path(nii_path)
            if not nii_path.exists():
                raise FeatureExtractionError(
                    f"MRI file not found: {nii_path}"
                )
            
            # Load atlas if not already loaded
            if self._atlas_img is None:
                self.load_atlas()
            
            # Ensure MNI space if requested
            if ensure_mni:
                img_to_process = self._ensure_mni_space(str(nii_path))
            else:
                import nibabel as nib
                img_to_process = nib.load(str(nii_path))
            
            # Validate ROI list
            self._validate_roi_list(roi_list)
            
            # Get ROI indices
            roi_mapping = self.get_roi_mapping()
            roi_indices = [roi_mapping[roi] for roi in roi_list]
            
            # Create masker if not exists or if ROI list changed
            if self._masker is None:
                print("Creating NiftiLabelsMasker...")
                self._masker = NiftiLabelsMasker(
                    labels_img=self._atlas_img,
                    standardize=standardize,
                    strategy='mean',  # Extract mean intensity per ROI
                    verbose=0
                )
                self._masker.fit()
                print("✓ Masker fitted")
            
            # Extract features for all ROIs
            print("Extracting features...")
            all_features = self._masker.transform(img_to_process)
            
            # Select only the ROIs we need
            # Note: masker returns features for all atlas regions
            # We need to select the specific indices
            
            # Handle both 1D and 2D cases
            if all_features.ndim == 2:
                # 2D case: (n_samples=1, n_regions)
                selected_features = all_features[0, [idx - 1 for idx in roi_indices]]
            else:
                # 1D case: (n_regions,)
                selected_features = all_features[[idx - 1 for idx in roi_indices]]
            
            # Validate output shape
            expected_shape = (len(roi_list),)
            if selected_features.shape != expected_shape:
                raise FeatureExtractionError(
                    f"Expected feature shape {expected_shape}, "
                    f"got {selected_features.shape}"
                )
            
            print(f"✓ Extracted {len(selected_features)} features")
            print(f"  - Feature range: [{selected_features.min():.3f}, {selected_features.max():.3f}]")
            print(f"  - Feature mean: {selected_features.mean():.3f}")
            print("=" * 35 + "\n")
            
            return selected_features
            
        except Exception as e:
            if isinstance(e, (FeatureExtractionError, AtlasLoadError)):
                raise
            raise FeatureExtractionError(
                f"Failed to extract features: {e}. "
                f"Please ensure the input is a valid T1-weighted MRI in NIfTI format."
            )
    
    def get_feature_dict(
        self,
        nii_path: str,
        roi_list: List[str]
    ) -> Dict[str, float]:
        """
        Extract features and return as dictionary
        
        Args:
            nii_path: Path to T1 MRI NIfTI file
            roi_list: List of ROI names
            
        Returns:
            Dictionary mapping ROI names to their feature values
        """
        features = self.extract_features(nii_path, roi_list)
        return dict(zip(roi_list, features))
    
    def clear_cache(self):
        """Clear cached atlas and masker"""
        self._atlas_img = None
        self._atlas_labels = None
        self._masker = None
        self._roi_mapping = None
        print("✓ Feature extractor cache cleared")
