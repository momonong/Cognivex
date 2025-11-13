#!/bin/bash
# 清理未使用程式碼的腳本
# 執行前請確保已經備份！

set -e  # 遇到錯誤立即停止

echo "========================================="
echo "  Cognivex 未使用程式碼清理腳本"
echo "========================================="
echo ""

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 檢查是否在專案根目錄
if [ ! -f "app.py" ]; then
    echo -e "${RED}錯誤: 請在專案根目錄執行此腳本${NC}"
    exit 1
fi

# 步驟 1: 確認備份
echo -e "${YELLOW}步驟 1: 檢查 Git 狀態${NC}"
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}警告: 有未提交的變更！${NC}"
    echo "請先提交或暫存您的變更："
    echo "  git add -A"
    echo "  git commit -m 'Backup before cleanup'"
    read -p "是否繼續？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 創建備份標籤
echo "創建備份標籤..."
git tag -f backup-before-cleanup-$(date +%Y%m%d-%H%M%S)
echo -e "${GREEN}✓ 備份標籤已創建${NC}"
echo ""

# 步驟 2: 創建 archive 目錄
echo -e "${YELLOW}步驟 2: 創建 archive 目錄結構${NC}"
mkdir -p archive/old_pipelines
mkdir -p archive/data_scripts
mkdir -p archive/exploratory_tests
mkdir -p archive/ui_backups
mkdir -p archive/placeholder_files
echo -e "${GREEN}✓ Archive 目錄已創建${NC}"
echo ""

# 步驟 3: 移動檔案
echo -e "${YELLOW}步驟 3: 移動未使用的檔案到 archive/${NC}"

# 3.1 UI 備份檔案
if [ -f "app/ui/structural_mri_components_backup.py" ]; then
    echo "移動 UI 備份檔案..."
    mv app/ui/structural_mri_components_backup.py archive/ui_backups/
    echo -e "${GREEN}✓ structural_mri_components_backup.py${NC}"
fi

# 3.2 舊 pipeline 系統
if [ -d "app/core/fmri_processing/pipelines" ]; then
    echo "移動舊 pipeline 系統..."
    mv app/core/fmri_processing/pipelines archive/old_pipelines/
    echo -e "${GREEN}✓ pipelines/ 目錄${NC}"
fi

# 3.3 fmri_model_loader.py (佔位符)
if [ -f "app/core/fmri_processing/fmri_model_loader.py" ]; then
    echo "移動 fmri_model_loader.py..."
    mv app/core/fmri_processing/fmri_model_loader.py archive/placeholder_files/
    echo -e "${GREEN}✓ fmri_model_loader.py${NC}"
fi

# 3.4 資料處理腳本
echo "移動資料處理腳本..."
for script in copy_smri_data.py create_mock_model.py reorganize_data.py rename_data_folders.py; do
    if [ -f "scripts/$script" ]; then
        mv "scripts/$script" archive/data_scripts/
        echo -e "${GREEN}✓ $script${NC}"
    fi
done

# 3.5 探索性測試檔案
echo "移動探索性測試檔案..."
cd tests
for file in analyze_vol_act.py brain_region.py capsnet_info.py check_*.py find_t.py \
            gpt.py image_explain.py model_info.py nii_*.py ollama.py prototype.py \
            region_network_map.py update_map_csv.py vertex*.py; do
    if [ -f "$file" ]; then
        mv "$file" ../archive/exploratory_tests/
        echo -e "${GREEN}✓ $file${NC}"
    fi
done
cd ..

# 3.6 model 目錄中的評估腳本
if [ -f "model/evaluate_model_accuracy.py" ]; then
    echo "移動模型評估腳本..."
    mv model/evaluate_model_accuracy.py archive/data_scripts/
    echo -e "${GREEN}✓ evaluate_model_accuracy.py${NC}"
fi

echo ""
echo -e "${GREEN}✓ 所有檔案已移動到 archive/${NC}"
echo ""

# 步驟 4: 創建 archive README
echo -e "${YELLOW}步驟 4: 創建 archive 說明文件${NC}"
cat > archive/README.md << 'EOF'
# Archive 目錄

此目錄包含已從主要程式碼庫中移除但保留以供參考的檔案。

## 目錄結構

- **old_pipelines/**: 舊版 fMRI 處理 pipeline 系統
  - 已被 `generic_pipeline_steps.py` 取代
  - 保留以供參考舊實作方式

- **data_scripts/**: 一次性資料處理腳本
  - 用於資料遷移和重組
  - 通常只執行一次

- **exploratory_tests/**: 探索性和除錯測試檔案
  - 開發過程中的臨時測試
  - 不是正式測試套件的一部分

- **ui_backups/**: UI 組件的備份檔案
  - 開發過程中的備份版本

- **placeholder_files/**: 佔位符檔案
  - 未完成或空實作的檔案

## 恢復檔案

如需恢復任何檔案，可以從此目錄複製回原位置，或使用 git 歷史記錄。

## 清理日期

檔案移至此目錄的日期: $(date +%Y-%m-%d)

## Git 備份標籤

在清理前創建的備份標籤: backup-before-cleanup-*
EOF

echo -e "${GREEN}✓ Archive README 已創建${NC}"
echo ""

# 步驟 5: 清理 __pycache__
echo -e "${YELLOW}步驟 5: 清理 __pycache__ 目錄${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✓ __pycache__ 已清理${NC}"
echo ""

# 步驟 6: 顯示摘要
echo "========================================="
echo -e "${GREEN}清理完成！${NC}"
echo "========================================="
echo ""
echo "已移動的檔案："
echo "  - UI 備份: 1 個檔案"
echo "  - 舊 pipeline: 1 個目錄"
echo "  - 資料腳本: ~5 個檔案"
echo "  - 測試檔案: ~20 個檔案"
echo ""
echo "下一步："
echo "  1. 執行測試確認系統正常:"
echo "     python -m pytest tests/test_*.py"
echo ""
echo "  2. 測試 Streamlit 應用:"
echo "     streamlit run app.py"
echo ""
echo "  3. 如果一切正常，提交變更:"
echo "     git add -A"
echo "     git commit -m 'Clean up unused code - moved to archive/'"
echo ""
echo "  4. 如需恢復，使用備份標籤:"
echo "     git tag -l 'backup-before-cleanup-*'"
echo "     git checkout <tag-name>"
echo ""
echo -e "${YELLOW}注意: archive/ 目錄已加入 git，可以隨時恢復檔案${NC}"
echo ""
