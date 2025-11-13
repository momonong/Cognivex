"""
3D Patch Extraction from Multi-modal MRI using AAL-116 ROI Masks
使用 AAL-116 ROI Masks 從多模態 MRI 提取 3D Patches
"""

import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from nilearn import datasets, image as nimg
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AAL116PatchExtractor:
    """
    Extract 3D patches from multi-modal MRI using AAL-116 atlas
    
    Workflow:
    1. Load AAL-116 atlas (116 ROI masks)
    2. For each ROI:
       - Extract bounding box
       - Crop 3D patch from each modality (T1, T2-FLAIR, DWI)
       - Resize to target size (32×32×32)
    3. Return 116 patches per modality
    """
    
    def __init__(
        self,
        target_patch_size=(32, 32, 32),
        padding=2,
        min_patch_size=(8, 8, 8),
        device='cpu'
    ):
        """
        Parameters:
        -----------
        target_patch_size : tuple
            Target size for all patches (D, H, W)
        padding : int
            Padding around ROI bounding box (in voxels)
        min_patch_size : tuple
            Minimum patch size (D, H, W)
        device : str
            Device for torch tensors ('cpu' or 'cuda')
        """
        self.target_patch_size = target_patch_size
        self.padding = padding
        self.min_patch_size = min_patch_size
        self.device = device
        
        # Load AAL atlas
        self.atlas_img, self.atlas_data, self.roi_labels = self._load_aal_atlas()
        
        print(f"[OK] AAL-116 Patch Extractor initialized")
        print(f"   Target patch size: {target_patch_size}")
        print(f"   Number of ROIs: {len(self.roi_labels)}")
    
    def _load_aal_atlas(self):
        """Load AAL-116 atlas"""
        try:
            # Try to load from nilearn
            aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
            atlas_img = nimg.load_img(aal_atlas.maps)
            atlas_data = atlas_img.get_fdata().astype(np.int16)
            
            # Get ROI labels (excluding background)
            roi_labels = [label.decode('utf-8') if isinstance(label, bytes) else label 
                         for label in aal_atlas.labels]
            
            # Remove 'Background' if present
            if roi_labels[0].lower() == 'background':
                roi_labels = roi_labels[1:]
            
            print(f"[OK] Loaded AAL atlas from nilearn")
            print(f"   Atlas shape: {atlas_data.shape}")
            print(f"   Number of ROIs: {len(roi_labels)}")
            print(f"   ROI indices: {np.unique(atlas_data)[1:][:5]}... (showing first 5)")
            
            return atlas_img, atlas_data, roi_labels
            
        except Exception as e:
            print(f"[WARN] Could not load AAL atlas from nilearn: {e}")
            print("   Please ensure nilearn is installed and has internet access")
            raise
    
    def _get_roi_bounding_box(self, roi_mask):
        """
        Get bounding box of ROI with padding
        
        Parameters:
        -----------
        roi_mask : np.ndarray
            Binary mask of ROI
        
        Returns:
        --------
        bbox : tuple
            Bounding box coordinates (z_min, z_max, y_min, y_max, x_min, x_max)
        """
        # Find non-zero coordinates
        coords = np.argwhere(roi_mask > 0)
        
        if len(coords) == 0:
            return None
        
        # Get min/max coordinates
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        
        # Add padding
        z_min = max(0, z_min - self.padding)
        y_min = max(0, y_min - self.padding)
        x_min = max(0, x_min - self.padding)
        
        z_max = min(roi_mask.shape[0], z_max + self.padding + 1)
        y_max = min(roi_mask.shape[1], y_max + self.padding + 1)
        x_max = min(roi_mask.shape[2], x_max + self.padding + 1)
        
        return (z_min, z_max, y_min, y_max, x_min, x_max)
    
    def _resize_patch(self, patch):
        """
        Resize 3D patch to target size using trilinear interpolation
        
        Parameters:
        -----------
        patch : np.ndarray
            3D patch of shape (D, H, W)
        
        Returns:
        --------
        resized_patch : np.ndarray
            Resized patch of shape target_patch_size
        """
        if patch.shape == self.target_patch_size:
            return patch
        
        # Calculate zoom factors
        zoom_factors = [
            self.target_patch_size[i] / patch.shape[i]
            for i in range(3)
        ]
        
        # Resize using scipy
        resized_patch = ndimage.zoom(patch, zoom_factors, order=1)  # order=1: trilinear
        
        return resized_patch
    
    def extract_patches_from_subject(self, t1_path, t2_path, dwi_path):
        """
        Extract 116 patches from each modality for one subject
        
        Parameters:
        -----------
        t1_path : str or Path
            Path to T1 image
        t2_path : str or Path
            Path to T2-FLAIR image
        dwi_path : str or Path
            Path to DWI image
        
        Returns:
        --------
        patches : dict
            Dictionary with keys 'T1', 'T2_FLAIR', 'DWI'
            Each value is a torch.Tensor of shape (116, 1, D, H, W)
        """
        # Load images
        t1_img = nimg.load_img(str(t1_path))
        t2_img = nimg.load_img(str(t2_path))
        dwi_img = nimg.load_img(str(dwi_path))
        
        # Resample atlas to match T1 image space
        atlas_resampled = nimg.resample_to_img(
            self.atlas_img, t1_img, interpolation='nearest'
        )
        atlas_data = atlas_resampled.get_fdata().astype(np.int16)
        
        # Get image data
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()
        dwi_data = dwi_img.get_fdata()
        
        # Initialize patch lists
        t1_patches = []
        t2_patches = []
        dwi_patches = []
        
        # Extract patches for each ROI
        roi_indices = np.unique(atlas_data)[1:]  # Exclude background (0)
        
        for roi_idx in roi_indices:
            # Create ROI mask
            roi_mask = (atlas_data == roi_idx)
            
            # Get bounding box
            bbox = self._get_roi_bounding_box(roi_mask)
            
            if bbox is None:
                # Empty ROI, create zero patch
                zero_patch = np.zeros(self.target_patch_size)
                t1_patches.append(zero_patch)
                t2_patches.append(zero_patch)
                dwi_patches.append(zero_patch)
                continue
            
            z_min, z_max, y_min, y_max, x_min, x_max = bbox
            
            # Extract patches
            t1_patch = t1_data[z_min:z_max, y_min:y_max, x_min:x_max]
            t2_patch = t2_data[z_min:z_max, y_min:y_max, x_min:x_max]
            dwi_patch = dwi_data[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # Check minimum size
            if any(s < m for s, m in zip(t1_patch.shape, self.min_patch_size)):
                # Too small, create zero patch
                zero_patch = np.zeros(self.target_patch_size)
                t1_patches.append(zero_patch)
                t2_patches.append(zero_patch)
                dwi_patches.append(zero_patch)
                continue
            
            # Resize patches
            t1_patch = self._resize_patch(t1_patch)
            t2_patch = self._resize_patch(t2_patch)
            dwi_patch = self._resize_patch(dwi_patch)
            
            t1_patches.append(t1_patch)
            t2_patches.append(t2_patch)
            dwi_patches.append(dwi_patch)
        
        # Convert to torch tensors
        # Shape: (N_ROI, 1, D, H, W)
        t1_patches = torch.from_numpy(np.array(t1_patches)).float().unsqueeze(1)
        t2_patches = torch.from_numpy(np.array(t2_patches)).float().unsqueeze(1)
        dwi_patches = torch.from_numpy(np.array(dwi_patches)).float().unsqueeze(1)
        
        return {
            'T1': t1_patches.to(self.device),
            'T2_FLAIR': t2_patches.to(self.device),
            'DWI': dwi_patches.to(self.device)
        }
    
    def get_roi_labels(self):
        """Get list of ROI labels"""
        return self.roi_labels


def test_patch_extractor():
    """Test the patch extractor"""
    print("="*80)
    print("Testing AAL-116 Patch Extractor")
    print("="*80)
    
    # Initialize extractor
    extractor = AAL116PatchExtractor(
        target_patch_size=(32, 32, 32),
        padding=2,
        device='cpu'
    )
    
    # Test with sample data (if available)
    data_root = Path("E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI")
    
    if data_root.exists():
        # Find first NC subject
        nc_dir = data_root / "NC"
        if nc_dir.exists():
            t1_files = list(nc_dir.glob("*_T1.nii.gz"))
            
            if len(t1_files) > 0:
                t1_path = t1_files[0]
                base_name = str(t1_path).replace("_T1.nii.gz", "")
                t2_path = Path(base_name + "_T2_FLAIR.nii.gz")
                dwi_path = Path(base_name + "_DWI.nii.gz")
                
                if t2_path.exists() and dwi_path.exists():
                    print(f"\nTesting with subject: {t1_path.stem}")
                    
                    # Extract patches
                    patches = extractor.extract_patches_from_subject(
                        t1_path, t2_path, dwi_path
                    )
                    
                    print(f"\n[OK] Patch extraction successful!")
                    print(f"   T1 patches shape: {patches['T1'].shape}")
                    print(f"   T2 patches shape: {patches['T2_FLAIR'].shape}")
                    print(f"   DWI patches shape: {patches['DWI'].shape}")
                    print(f"   Expected: (116, 1, 32, 32, 32)")
                    
                    # Check statistics
                    print(f"\n   T1 patch statistics:")
                    print(f"     Min: {patches['T1'].min():.2f}")
                    print(f"     Max: {patches['T1'].max():.2f}")
                    print(f"     Mean: {patches['T1'].mean():.2f}")
                    
                    return
    
    print("\n[WARN] No test data found. Please check data path.")
    print("   Expected path: E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/")


if __name__ == "__main__":
    test_patch_extractor()
