import nibabel as nib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def check_time_dim(file_path):
    img = nib.load(file_path)
    return img.shape[-1]

def validate_time_dim(root_dir):
    time_dims = []
    root = Path(root_dir)
    
    for class_dir in ['AD', 'CN']:
        class_path = root / class_dir
        if not class_path.exists():
            logging.error(f"目錄不存在: {class_path}")
            continue
            
        for subj_dir in class_path.iterdir():
            if not subj_dir.is_dir():
                continue
                
            nii_files = list(subj_dir.glob('*.nii.gz'))
            if not nii_files:
                logging.warning(f"無.nii.gz文件: {subj_dir}")
                continue
                
            # 選擇最後修改時間最新的文件
            nii_file = max(nii_files, key=lambda f: f.stat().st_mtime)
            
            try:
                vol_num = check_time_dim(nii_file)
                time_dims.append(vol_num)
                logging.info(f"成功讀取: {nii_file}, volumes={vol_num}")
            except Exception as e:
                logging.error(f"文件讀取失敗: {nii_file}, 錯誤: {str(e)}")
    
    return sorted(list(set(time_dims)))

if __name__ == "__main__":
    unique_time_dims = validate_time_dim('data/raw')
    print(f"唯一時間軸長度: {unique_time_dims}")
