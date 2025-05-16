import os
import pandas as pd

def rename_files_by_csv(data_dir: str, csv_path: str):
    # 讀取對應表格
    df = pd.read_csv(csv_path, sep=',').dropna(subset=['AD_Subject', 'CN_Subject'])

    # 收集所有 AD 與 CN 的 subject ID
    ad_subjects = df['AD_Subject'].dropna().tolist()
    cn_subjects = df['CN_Subject'].dropna().tolist()

    # 處理資料夾內所有檔案
    files = os.listdir(data_dir)
    for fname in files:
        full_path = os.path.join(data_dir, fname)

        if not fname.endswith('.nii.gz'):
            print(f"[SKIP] Not a NIfTI file: {fname}")
            continue

        matched_id = None
        label = None

        for subject_id in ad_subjects:
            if subject_id in fname:
                matched_id = subject_id
                label = 'AD'
                break

        if not matched_id:
            for subject_id in cn_subjects:
                if subject_id in fname:
                    matched_id = subject_id
                    label = 'CN'
                    break

        if matched_id and label:
            # 組成新檔名
            new_name = fname.replace(matched_id, f"{label}_{matched_id}")
            new_path = os.path.join(data_dir, new_name)

            if not os.path.exists(new_path):
                os.rename(full_path, new_path)
                print(f"[RENAME] {fname} → {new_name}")
            else:
                print(f"[SKIP] already exists: {new_name}")
        else:
            print(f"[IGNORE] No match in CSV: {fname}")

if __name__ == '__main__':
    data_path = 'data/packages'
    csv_path = 'data/adcn_match.csv'  # ← 改成你的檔案路徑
    rename_files_by_csv(data_path, csv_path)
