import os

def rename_ad_files(data_dir: str):
    files = os.listdir(data_dir)
    for fname in files:
        if fname.endswith('_task-rest_bold.nii.gz') and 'ses-001' in fname:
            if '_AD_' not in fname:  # 避免重複改名
                new_name = fname.replace('ses-001', 'AD')
                old_path = os.path.join(data_dir, fname)
                new_path = os.path.join(data_dir, new_name)
                os.rename(old_path, new_path)
                print(f"[RENAME] {fname} → {new_name}")
            else:
                print(f"[SKIP] already renamed: {fname}")
        else:
            print(f"[IGNORE] {fname}")

if __name__ == '__main__':
    data_path = 'data/packages'
    rename_ad_files(data_path)
