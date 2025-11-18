import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchio as tio  # [新增] 強烈建議安裝 torchio 或是 monai 做 3D 增強

class CognivexDataset(Dataset):
    def __init__(self, csv_file, root_dir, classes=['NC', 'AD'], transform=None):
        """
        classes: 指定要包含的類別，例如 ['NC', 'AD'] 或 ['NC', 'MCI', 'AD']
        """
        self.raw_annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        
        # 動態建立 Label Mapping
        # 例如 ['NC', 'AD'] -> {'NC': 0, 'AD': 1}
        self.label_map = {name: i for i, name in enumerate(classes)}
        
        self.valid_subjects = self._filter_valid_subjects()

    def _filter_valid_subjects(self):
        valid_data = []
        print(f"[*] 正在篩選資料 (目標類別: {self.classes})...")
        
        for idx, row in self.raw_annotations.iterrows():
            group = row['diagnosis']
            
            # [新增] 只保留我們想要的類別
            if group not in self.classes:
                continue

            subject_id = row['new_id_base'].replace('_', '-')
            
            # 檢查檔案是否存在
            subject_dir = os.path.join(self.root_dir, group, subject_id)
            f1 = os.path.join(subject_dir, f"{subject_id}_GM_to_MNI.nii.gz")
            
            # 這裡只簡單檢查 GM，因為前處理腳本保證了如果成功就三個都有
            if os.path.exists(f1):
                valid_data.append(row)
        
        print(f"[*] 篩選完成: {len(valid_data)} 位受試者符合條件。")
        return pd.DataFrame(valid_data)

    def __len__(self):
        return len(self.valid_subjects)

    def __getitem__(self, idx):
        row = self.valid_subjects.iloc[idx]
        subject_id = row['new_id_base'].replace('_', '-')
        group = row['diagnosis']
        label = self.label_map[group]

        subject_dir = os.path.join(self.root_dir, group, subject_id)

        # 1. 讀取
        path_gm = os.path.join(subject_dir, f"{subject_id}_GM_to_MNI.nii.gz")
        path_fa = os.path.join(subject_dir, f"{subject_id}_FA_to_MNI.nii.gz")
        path_md = os.path.join(subject_dir, f"{subject_id}_MD_to_MNI.nii.gz")

        img_gm = nib.load(path_gm).get_fdata().astype(np.float32)
        img_fa = nib.load(path_fa).get_fdata().astype(np.float32)
        img_md = nib.load(path_md).get_fdata().astype(np.float32)

        # 2. 堆疊 (Channel, X, Y, Z)
        # 注意：這還是 numpy array
        volume = np.stack([img_gm, img_fa, img_md], axis=0)

        # 3. 正規化 (0-1)
        for c in range(3):
            v_min, v_max = volume[c].min(), volume[c].max()
            if v_max - v_min > 0:
                volume[c] = (volume[c] - v_min) / (v_max - v_min)
        
        # 4. [新增] 資料增強 (如果是 Tensor 輸入)
        # 如果有傳入 transform (通常是 torchio 或 monai)，在這裡套用
        # 為了簡單起見，這裡我們傳回 Tensor，在外部 DataLoader 做增強也可以
        
        return torch.from_numpy(volume), torch.tensor(label, dtype=torch.long)