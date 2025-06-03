import os
import numpy as np
import nibabel as nib
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class FMRI4DDataset(Dataset):
    def __init__(self, root_dir, window=5, stride=3):
        self.samples = []
        self.labels = []
        for class_label, class_name in enumerate(['AD', 'CN']):
            class_path = Path(root_dir) / class_name
            for subj_dir in class_path.iterdir():
                nii_files = list(subj_dir.glob('*.nii.gz'))
                if not nii_files:
                    continue
                nii_file = nii_files[0]
                img = nib.load(str(nii_file)).get_fdata()  # shape: (61,73,61,190)
                img = img[..., 3:]  # 去除前3個volumes -> (61,73,61,187)
                T = img.shape[-1]
                for start in range(0, T - window + 1, stride):
                    sample = img[..., start:start+window]  # (61,73,61,5)
                    sample = np.transpose(sample, (3,0,1,2))  # (5,61,73,61)
                    self.samples.append(sample)
                    self.labels.append(class_label)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.samples[idx][np.newaxis])  # (1,5,61,73,61)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
