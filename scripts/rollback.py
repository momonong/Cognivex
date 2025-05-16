import os

def undo_double_label(data_dir: str):
    files = os.listdir(data_dir)
    for fname in files:
        if not fname.endswith('.nii.gz'):
            continue

        # 處理 AD 重複：sub-AD_AD_xxx → sub-AD_xxx
        if 'sub-AD_AD_' in fname:
            new_name = fname.replace('sub-AD_AD_', 'sub-AD_')
        # 處理 CN 重複：sub-CN_CN_xxx → sub-CN_xxx
        elif 'sub-CN_CN_' in fname:
            new_name = fname.replace('sub-CN_CN_', 'sub-CN_')
        else:
            continue  # 不符合條件就跳過

        old_path = os.path.join(data_dir, fname)
        new_path = os.path.join(data_dir, new_name)

        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"[UNDO] {fname} → {new_name}")
        else:
            print(f"[SKIP] already exists: {new_name}")

if __name__ == '__main__':
    data_path = 'data/packages'
    undo_double_label(data_path)
