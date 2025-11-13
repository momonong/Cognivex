"""
Test the complete multi-modal ROI pipeline
測試完整的多模態 ROI Pipeline
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from resnet3d_mini import ResNet3D_Mini, MultiModalFeatureExtractor
from patch_extractor import AAL116PatchExtractor
from dataset import MultiModalROIDataset


def test_components():
    """Test all pipeline components"""
    print("="*80)
    print("Testing Multi-modal ROI Pipeline Components")
    print("="*80)
    
    # Test 1: 3D ResNet-10 Mini-CNN
    print("\n[1/5] Testing 3D ResNet-10 Mini-CNN...")
    try:
        model = ResNet3D_Mini(in_channels=1, feature_dim=64, initial_filters=32)
        x = torch.randn(2, 1, 32, 32, 32)
        features = model(x)
        
        assert features.shape == (2, 64), f"Expected (2, 64), got {features.shape}"
        print("[OK] 3D ResNet-10 Mini-CNN test passed")
        print(f"   Input: {x.shape} -> Output: {features.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")
        
    except Exception as e:
        print(f"[FAIL] 3D ResNet-10 Mini-CNN test failed: {e}")
        return False
    
    # Test 2: Multi-modal Feature Extractor
    print("\n[2/5] Testing Multi-modal Feature Extractor...")
    try:
        multi_model = MultiModalFeatureExtractor(feature_dim=64, initial_filters=32)
        
        t1_patches = torch.randn(2, 116, 1, 32, 32, 32)
        t2_patches = torch.randn(2, 116, 1, 32, 32, 32)
        dwi_patches = torch.randn(2, 116, 1, 32, 32, 32)
        
        features = multi_model(t1_patches, t2_patches, dwi_patches)
        
        expected_dim = 116 * 3 * 64  # 22,104
        assert features.shape == (2, expected_dim), f"Expected (2, {expected_dim}), got {features.shape}"
        print("[OK] Multi-modal Feature Extractor test passed")
        print(f"   Input: (2, 116, 1, 32, 32, 32) x 3 modalities")
        print(f"   Output: {features.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in multi_model.parameters())
        print(f"   Total parameters (3 Mini-CNNs): {total_params:,}")
        
    except Exception as e:
        print(f"[FAIL] Multi-modal Feature Extractor test failed: {e}")
        return False
    
    # Test 3: AAL-116 Patch Extractor
    print("\n[3/5] Testing AAL-116 Patch Extractor...")
    try:
        extractor = AAL116PatchExtractor(
            target_patch_size=(32, 32, 32),
            padding=2,
            device='cpu'
        )
        
        print("[OK] AAL-116 Patch Extractor initialized")
        print(f"   Number of ROIs: {len(extractor.roi_labels)}")
        print(f"   Target patch size: {extractor.target_patch_size}")
        
        # Test with real data if available
        data_root = Path(DATA_ROOT)
        if data_root.exists():
            nc_dir = data_root / "NC"
            if nc_dir.exists():
                t1_files = list(nc_dir.glob("*_T1.nii.gz"))
                
                if len(t1_files) > 0:
                    t1_path = t1_files[0]
                    base_name = str(t1_path).replace("_T1.nii.gz", "")
                    t2_path = Path(base_name + "_T2_FLAIR.nii.gz")
                    dwi_path = Path(base_name + "_DWI.nii.gz")
                    
                    if t2_path.exists() and dwi_path.exists():
                        print(f"   Testing with: {t1_path.stem}")
                        
                        patches = extractor.extract_patches_from_subject(
                            t1_path, t2_path, dwi_path
                        )
                        
                        assert patches['T1'].shape == (116, 1, 32, 32, 32)
                        assert patches['T2_FLAIR'].shape == (116, 1, 32, 32, 32)
                        assert patches['DWI'].shape == (116, 1, 32, 32, 32)
                        
                        print("[OK] Patch extraction test passed")
                        print(f"   T1 patches: {patches['T1'].shape}")
                        print(f"   T2 patches: {patches['T2_FLAIR'].shape}")
                        print(f"   DWI patches: {patches['DWI'].shape}")
                    else:
                        print("[WARN] Complete modalities not found, skipping real data test")
                else:
                    print("[WARN] No T1 files found, skipping real data test")
            else:
                print("[WARN] NC directory not found, skipping real data test")
        else:
            print("[WARN] Data root not found, skipping real data test")
        
    except Exception as e:
        print(f"[FAIL] AAL-116 Patch Extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Dataset
    print("\n[4/5] Testing Multi-modal ROI Dataset...")
    try:
        if Path(DATA_ROOT).exists():
            dataset = MultiModalROIDataset(
                data_root=DATA_ROOT,
                split='train',
                use_cache=False  # Don't use cache for testing
            )
            
            if len(dataset) > 0:
                print(f"[OK] Dataset initialized with {len(dataset)} subjects")
                
                # Test loading one sample
                sample = dataset[0]
                
                assert 'patches' in sample
                assert 'label' in sample
                assert 'subject_id' in sample
                
                print("[OK] Dataset test passed")
                print(f"   Subject ID: {sample['subject_id']}")
                print(f"   Label: {sample['label']}")
                print(f"   T1 patches: {sample['patches']['T1'].shape}")
            else:
                print("[WARN] Dataset is empty, skipping dataset test")
        else:
            print("[WARN] Data root not found, skipping dataset test")
        
    except Exception as e:
        print(f"[FAIL] Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: End-to-end forward pass
    print("\n[5/5] Testing end-to-end forward pass...")
    try:
        # Create dummy data
        batch_size = 2
        n_rois = 116
        patch_size = (32, 32, 32)
        
        t1_patches = torch.randn(batch_size, n_rois, 1, *patch_size)
        t2_patches = torch.randn(batch_size, n_rois, 1, *patch_size)
        dwi_patches = torch.randn(batch_size, n_rois, 1, *patch_size)
        
        # Forward pass
        model = MultiModalFeatureExtractor(feature_dim=64, initial_filters=32)
        model.eval()
        
        with torch.no_grad():
            features = model(t1_patches, t2_patches, dwi_patches)
        
        # Check output
        expected_dim = n_rois * 3 * 64
        assert features.shape == (batch_size, expected_dim)
        
        print("[OK] End-to-end forward pass test passed")
        print(f"   Batch size: {batch_size}")
        print(f"   Input: {n_rois} ROIs × 3 modalities × {patch_size}")
        print(f"   Output: {features.shape} ({expected_dim} features)")
        
        # Check feature statistics
        print(f"   Feature statistics:")
        print(f"     Mean: {features.mean():.4f}")
        print(f"     Std:  {features.std():.4f}")
        print(f"     Min:  {features.min():.4f}")
        print(f"     Max:  {features.max():.4f}")
        
    except Exception as e:
        print(f"[FAIL] End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def print_system_info():
    """Print system information"""
    print("\n" + "="*80)
    print("System Information")
    print("="*80)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Data root: {DATA_ROOT}")
    print(f"  Model dir: {MODEL_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of ROIs: {NUM_ROIS}")
    print(f"  Feature dimension: {TOTAL_FEATURE_DIM}")


def main():
    """Main test function"""
    print_system_info()
    
    success = test_components()
    
    print("\n" + "="*80)
    if success:
        print("[SUCCESS] All tests passed!")
        print("="*80)
        print("\nNext steps:")
        print("1. Prepare your data in the correct format")
        print("2. Run: python scripts/multimodal_roi/train.py")
        print("3. After training, run: python scripts/multimodal_roi/inference.py")
    else:
        print("[FAIL] Some tests failed!")
        print("="*80)
        print("\nPlease check the error messages above and fix the issues.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
