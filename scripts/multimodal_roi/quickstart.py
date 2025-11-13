"""
Quick Start Script for Multi-modal ROI Pipeline
多模態 ROI Pipeline 快速啟動腳本

這個腳本會引導你完成整個 pipeline 的設置和測試。
"""

import sys
from pathlib import Path
import subprocess

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(text)
    print("="*80 + "\n")

def print_step(step_num, total_steps, text):
    """Print a step indicator"""
    print(f"\n[步驟 {step_num}/{total_steps}] {text}")
    print("-"*80)

def check_dependencies():
    """Check if all required packages are installed"""
    print_step(1, 5, "檢查依賴套件")
    
    required_packages = [
        'torch',
        'nibabel',
        'nilearn',
        'sklearn',
        'xgboost',
        'pandas',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} - 未安裝")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n[WARN] 缺少以下套件: {', '.join(missing_packages)}")
        print("\n請執行以下命令安裝:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n[OK] 所有依賴套件已安裝")
    return True

def check_data():
    """Check if data is available"""
    print_step(2, 5, "檢查數據")
    
    from config import DATA_ROOT
    
    data_root = Path(DATA_ROOT)
    
    if not data_root.exists():
        print(f"[FAIL] 數據目錄不存在: {data_root}")
        print("\n請確保數據位於正確的位置:")
        print(f"  {data_root}/")
        print(f"  +-- NC/")
        print(f"  |   +-- *_T1.nii.gz")
        print(f"  |   +-- *_T2_FLAIR.nii.gz")
        print(f"  |   +-- *_DWI.nii.gz")
        print(f"  +-- MCI/")
        print(f"  +-- AD/")
        return False
    
    # Check for each class
    classes = ['NC', 'MCI', 'AD']
    total_subjects = 0
    
    for class_name in classes:
        class_dir = data_root / class_name
        if class_dir.exists():
            t1_files = list(class_dir.glob("*_T1.nii.gz"))
            
            # Check for complete modalities
            complete = 0
            for t1_path in t1_files:
                base_name = str(t1_path).replace("_T1.nii.gz", "")
                t2_path = Path(base_name + "_T2_FLAIR.nii.gz")
                dwi_path = Path(base_name + "_DWI.nii.gz")
                
                if t2_path.exists() and dwi_path.exists():
                    complete += 1
            
            print(f"[OK] {class_name}: {complete} 個完整受試者 (共 {len(t1_files)} 個 T1 檔案)")
            total_subjects += complete
        else:
            print(f"[WARN] {class_name}: 目錄不存在")
    
    if total_subjects == 0:
        print("\n[FAIL] 沒有找到完整的受試者數據")
        return False
    
    print(f"\n[OK] 總共找到 {total_subjects} 個完整的受試者")
    
    if total_subjects < 30:
        print("[WARN] 警告: 樣本數量較少，可能影響模型效能")
    
    return True

def run_tests():
    """Run pipeline tests"""
    print_step(3, 5, "測試 Pipeline 組件")
    
    print("執行測試腳本...")
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/multimodal_roi/test_pipeline.py"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n[OK] 所有測試通過")
            return True
        else:
            print("\n[FAIL] 測試失敗")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\n[FAIL] 執行測試時發生錯誤: {e}")
        return False

def show_training_options():
    """Show training options"""
    print_step(4, 5, "訓練選項")
    
    print("你可以選擇以下訓練方式:\n")
    
    print("選項 1: 完整訓練 (推薦)")
    print("  - 訓練 3 個 Mini-CNNs + XGBoost")
    print("  - 預計時間: 2-4 小時 (GPU) / 8-12 小時 (CPU)")
    print("  - 命令: python scripts/multimodal_roi/train.py")
    
    print("\n選項 2: 快速測試訓練")
    print("  - 使用較少的 epochs 進行測試")
    print("  - 預計時間: 30-60 分鐘")
    print("  - 需要修改 config.py: NUM_EPOCHS = 10")
    
    print("\n選項 3: 使用預訓練模型")
    print("  - 如果有預訓練模型，直接進行推理")
    print("  - 命令: python scripts/multimodal_roi/inference.py")

def show_next_steps():
    """Show next steps"""
    print_step(5, 5, "下一步")
    
    print("建議的工作流程:\n")
    
    print("1. 開始訓練:")
    print("   python scripts/multimodal_roi/train.py")
    
    print("\n2. 監控訓練 (可選):")
    print("   tensorboard --logdir output/multimodal_roi/logs")
    
    print("\n3. 訓練完成後，進行推理:")
    print("   python scripts/multimodal_roi/inference.py")
    
    print("\n4. 查看結果:")
    print("   - 模型: model/multimodal_roi/")
    print("   - 訓練歷史: output/multimodal_roi/training_history.csv")
    print("   - 特徵重要性: output/multimodal_roi/feature_importance.csv")
    
    print("\n5. 詳細文檔:")
    print("   - 使用指南: scripts/multimodal_roi/README.md")
    print("   - 優化方案: docs/MULTIMODAL_ROI_OPTIMIZATION.md")

def main():
    """Main function"""
    print_header("多模態 ROI Pipeline 快速啟動")
    
    print("這個腳本會幫助你設置和測試完整的 pipeline。")
    print("請確保你已經閱讀了 docs/MULTIMODAL_ROI_OPTIMIZATION.md")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n[FAIL] 請先安裝缺少的套件")
        return False
    
    # Step 2: Check data
    if not check_data():
        print("\n[FAIL] 請先準備數據")
        return False
    
    # Step 3: Run tests
    print("\n是否要運行測試? (y/n): ", end='')
    response = input().strip().lower()
    
    if response == 'y':
        if not run_tests():
            print("\n[WARN] 測試失敗，但你仍然可以繼續")
    
    # Step 4: Show training options
    show_training_options()
    
    # Step 5: Show next steps
    show_next_steps()
    
    print_header("設置完成")
    
    print("[SUCCESS] 你已經準備好開始訓練了！")
    print("\n建議執行:")
    print("  python scripts/multimodal_roi/train.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[WARN] 用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[FAIL] 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
