"""
Dataset for Multi-modal ROI Feature Extraction
多模態 ROI 特徵提取數據集
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from patch_extractor import AAL116PatchExtractor
from config import *


class MultiModalROIDataset(Dataset):
    """
    Dataset for multi-modal MRI with ROI patches
    
    Features:
    - Lazy loading: Extract patches on-the-fly or load from cache
    - Multi-modal: T1, T2-FLAIR, DWI
    - 116 ROI patches per modality
    """
    
    def __init__(
        self,
        data_root,
        split='train',
        use_cache=True,
        cache_dir=None,
        transform=None
    ):
        """
        Parameters:
        -----------
        data_root : str or Path
            Root directory containing NC/MCI/AD folders
        split : str
            'train', 'val', or 'test'
        use_cache : bool
            Whether to use cached patches
        cache_dir : str or Path
            Directory to store cached patches
        transform : callable
            Optional transform to apply to patches
        """
        self.data_root = Path(data_root)
        self.split = split
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / split
        self.transform = transform
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize patch extractor
        self.patch_extractor = AAL116PatchExtractor(
            target_patch_size=PATCH_CONFIG['target_patch_size'],
            padding=PATCH_CONFIG['padding'],
            min_patch_size=PATCH_CONFIG['min_patch_size'],
            device='cpu'  # Extract on CPU, move to GPU during training
        )
        
        # Collect all subjects
        self.subjects = self._collect_subjects()
        
        print(f"[OK] Dataset initialized: {split}")
        print(f"   Total subjects: {len(self.subjects)}")
        print(f"   NC: {sum(1 for s in self.subjects if s['label'] == 0)}")
        print(f"   MCI: {sum(1 for s in self.subjects if s['label'] == 1)}")
        print(f"   AD: {sum(1 for s in self.subjects if s['label'] == 2)}")
    
    def _collect_subjects(self):
        """Collect all subjects with complete modalities"""
        subjects = []
        
        for label_name, label_id in LABEL_MAP.items():
            class_dir = self.data_root / label_name
            
            if not class_dir.exists():
                print(f"[WARN] Directory not found: {class_dir}")
                continue
            
            # Find all T1 files
            t1_files = sorted(list(class_dir.glob("*_T1.nii.gz")))
            
            for t1_path in t1_files:
                # Check for corresponding T2 and DWI files
                base_name = str(t1_path).replace("_T1.nii.gz", "")
                t2_path = Path(base_name + "_T2_FLAIR.nii.gz")
                dwi_path = Path(base_name + "_DWI.nii.gz")
                
                if t2_path.exists() and dwi_path.exists():
                    subject_id = t1_path.stem.replace("_T1", "")
                    
                    subjects.append({
                        'subject_id': subject_id,
                        'label': label_id,
                        'label_name': label_name,
                        't1_path': t1_path,
                        't2_path': t2_path,
                        'dwi_path': dwi_path,
                        'cache_path': self.cache_dir / f"{subject_id}.pkl"
                    })
        
        return subjects
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        """
        Get one subject's data
        
        Returns:
        --------
        patches : dict
            Dictionary with keys 'T1', 'T2_FLAIR', 'DWI'
            Each value is a tensor of shape (116, 1, D, H, W)
        label : int
            Class label (0: NC, 1: MCI, 2: AD)
        subject_id : str
            Subject identifier
        """
        subject = self.subjects[idx]
        
        # Try to load from cache
        if self.use_cache and subject['cache_path'].exists():
            try:
                with open(subject['cache_path'], 'rb') as f:
                    patches = pickle.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load cache for {subject['subject_id']}: {e}")
                patches = None
        else:
            patches = None
        
        # Extract patches if not cached
        if patches is None:
            patches = self.patch_extractor.extract_patches_from_subject(
                subject['t1_path'],
                subject['t2_path'],
                subject['dwi_path']
            )
            
            # Save to cache
            if self.use_cache:
                try:
                    with open(subject['cache_path'], 'wb') as f:
                        pickle.dump(patches, f)
                except Exception as e:
                    print(f"[WARN] Failed to save cache for {subject['subject_id']}: {e}")
        
        # Apply transform if provided
        if self.transform:
            patches = self.transform(patches)
        
        return {
            'patches': patches,
            'label': subject['label'],
            'subject_id': subject['subject_id']
        }
    
    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        labels = [s['label'] for s in self.subjects]
        class_counts = np.bincount(labels)
        total = len(labels)
        
        # Inverse frequency weighting
        weights = total / (len(class_counts) * class_counts)
        
        return torch.FloatTensor(weights)


def create_dataloaders(
    data_root,
    batch_size=4,
    num_workers=4,
    use_cache=True,
    train_split=0.7,
    val_split=0.15,
    random_seed=42
):
    """
    Create train/val/test dataloaders with stratified split
    
    Parameters:
    -----------
    data_root : str or Path
        Root directory containing NC/MCI/AD folders
    batch_size : int
        Batch size
    num_workers : int
        Number of workers for data loading
    use_cache : bool
        Whether to use cached patches
    train_split : float
        Proportion of training data
    val_split : float
        Proportion of validation data
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dataloaders : dict
        Dictionary with keys 'train', 'val', 'test'
    """
    from sklearn.model_selection import train_test_split
    
    # Create full dataset to get all subjects
    full_dataset = MultiModalROIDataset(
        data_root=data_root,
        split='full',
        use_cache=use_cache
    )
    
    # Get all subjects and labels
    subjects = full_dataset.subjects
    labels = [s['label'] for s in subjects]
    
    # Stratified split
    train_idx, temp_idx = train_test_split(
        range(len(subjects)),
        test_size=(1 - train_split),
        stratify=labels,
        random_state=random_seed
    )
    
    temp_labels = [labels[i] for i in temp_idx]
    val_size = val_split / (1 - train_split)
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=random_seed
    )
    
    # Create subset datasets
    train_subjects = [subjects[i] for i in train_idx]
    val_subjects = [subjects[i] for i in val_idx]
    test_subjects = [subjects[i] for i in test_idx]
    
    # Create datasets
    train_dataset = MultiModalROIDataset(
        data_root=data_root,
        split='train',
        use_cache=use_cache
    )
    train_dataset.subjects = train_subjects
    
    val_dataset = MultiModalROIDataset(
        data_root=data_root,
        split='val',
        use_cache=use_cache
    )
    val_dataset.subjects = val_subjects
    
    test_dataset = MultiModalROIDataset(
        data_root=data_root,
        split='test',
        use_cache=use_cache
    )
    test_dataset.subjects = test_subjects
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("\n" + "="*60)
    print("DataLoaders created:")
    print(f"  Train: {len(train_dataset)} subjects")
    print(f"  Val:   {len(val_dataset)} subjects")
    print(f"  Test:  {len(test_dataset)} subjects")
    print("="*60)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def test_dataset():
    """Test the dataset"""
    print("="*80)
    print("Testing Multi-Modal ROI Dataset")
    print("="*80)
    
    # Create dataset
    dataset = MultiModalROIDataset(
        data_root=DATA_ROOT,
        split='train',
        use_cache=True
    )
    
    if len(dataset) == 0:
        print("[WARN] No subjects found in dataset")
        return
    
    # Test loading one sample
    print(f"\nLoading first sample...")
    sample = dataset[0]
    
    print(f"\n[OK] Sample loaded successfully!")
    print(f"   Subject ID: {sample['subject_id']}")
    print(f"   Label: {sample['label']}")
    print(f"   T1 patches shape: {sample['patches']['T1'].shape}")
    print(f"   T2 patches shape: {sample['patches']['T2_FLAIR'].shape}")
    print(f"   DWI patches shape: {sample['patches']['DWI'].shape}")
    
    # Test dataloader
    print(f"\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    batch = next(iter(loader))
    print(f"\n[OK] Batch loaded successfully!")
    print(f"   Batch size: {len(batch['label'])}")
    print(f"   T1 patches shape: {batch['patches']['T1'].shape}")
    print(f"   Labels: {batch['label']}")


if __name__ == "__main__":
    test_dataset()
