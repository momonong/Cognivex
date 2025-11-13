@echo off
REM 清理未使用程式碼的腳本 (Windows 版本 - 修正版)
REM 執行前請確保已經備份！
REM 
REM 重要修正: pipelines/ 目錄仍在使用中，不會被移動！

setlocal enabledelayedexpansion

echo =========================================
echo   Cognivex 未使用程式碼清理腳本
echo   (修正版 - 保留 pipelines 目錄)
echo =========================================
echo.

REM 檢查是否在專案根目錄
if not exist "app.py" (
    echo [錯誤] 請在專案根目錄執行此腳本
    exit /b 1
)

REM 步驟 1: 確認備份
echo [步驟 1] 檢查 Git 狀態
git status --porcelain > nul 2>&1
if errorlevel 1 (
    echo [警告] Git 未初始化或不可用
    pause
)

echo 創建備份標籤...
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
git tag backup-before-cleanup-%mydate%-%mytime%
echo [完成] 備份標籤已創建
echo.

REM 步驟 2: 創建 archive 目錄
echo [步驟 2] 創建 archive 目錄結構
if not exist "archive" mkdir archive
if not exist "archive\data_scripts" mkdir archive\data_scripts
if not exist "archive\exploratory_tests" mkdir archive\exploratory_tests
if not exist "archive\ui_backups" mkdir archive\ui_backups
if not exist "archive\placeholder_files" mkdir archive\placeholder_files
echo [完成] Archive 目錄已創建
echo.

REM 步驟 3: 移動檔案
echo [步驟 3] 移動未使用的檔案到 archive\
echo.
echo [重要] pipelines\ 目錄仍在使用中，將被保留！
echo.

REM 3.1 UI 備份檔案
if exist "app\ui\structural_mri_components_backup.py" (
    echo 移動 UI 備份檔案...
    move "app\ui\structural_mri_components_backup.py" "archive\ui_backups\" >nul
    echo [完成] structural_mri_components_backup.py
)

REM 3.2 fmri_model_loader.py (佔位符)
if exist "app\core\fmri_processing\fmri_model_loader.py" (
    echo 移動 fmri_model_loader.py...
    move "app\core\fmri_processing\fmri_model_loader.py" "archive\placeholder_files\" >nul
    echo [完成] fmri_model_loader.py
)

REM 3.3 資料處理腳本
echo 移動資料處理腳本...
if exist "scripts\copy_smri_data.py" move "scripts\copy_smri_data.py" "archive\data_scripts\" >nul
if exist "scripts\create_mock_model.py" move "scripts\create_mock_model.py" "archive\data_scripts\" >nul
if exist "scripts\reorganize_data.py" move "scripts\reorganize_data.py" "archive\data_scripts\" >nul
if exist "scripts\rename_data_folders.py" move "scripts\rename_data_folders.py" "archive\data_scripts\" >nul
echo [完成] 資料處理腳本

REM 3.4 探索性測試檔案
echo 移動探索性測試檔案...
cd tests
for %%f in (analyze_vol_act.py brain_region.py capsnet_info.py check_*.py find_t.py gpt.py image_explain.py model_info.py nii_*.py ollama.py prototype.py region_network_map.py update_map_csv.py vertex*.py) do (
    if exist "%%f" (
        move "%%f" "..\archive\exploratory_tests\" >nul 2>&1
    )
)
cd ..
echo [完成] 探索性測試檔案

REM 3.5 model 目錄中的評估腳本
if exist "model\evaluate_model_accuracy.py" (
    echo 移動模型評估腳本...
    move "model\evaluate_model_accuracy.py" "archive\data_scripts\" >nul
    echo [完成] evaluate_model_accuracy.py
)

echo.
echo [完成] 所有檔案已移動到 archive\
echo [重要] pipelines\ 目錄已保留（仍在使用中）
echo.

REM 步驟 4: 創建 archive README
echo [步驟 4] 創建 archive 說明文件
(
echo # Archive 目錄
echo.
echo 此目錄包含已從主要程式碼庫中移除但保留以供參考的檔案。
echo.
echo ## 重要說明
echo.
echo **pipelines/ 目錄未被移動**: 經過確認，`app/core/fmri_processing/pipelines/` 
echo 目錄仍被 `generic_pipeline_steps.py` 使用，因此保留在原位置。
echo.
echo ## 目錄結構
echo.
echo - **data_scripts/**: 一次性資料處理腳本
echo   - 用於資料遷移和重組
echo   - 通常只執行一次
echo.
echo - **exploratory_tests/**: 探索性和除錯測試檔案
echo   - 開發過程中的臨時測試
echo   - 不是正式測試套件的一部分
echo.
echo - **ui_backups/**: UI 組件的備份檔案
echo   - 開發過程中的備份版本
echo.
echo - **placeholder_files/**: 佔位符檔案
echo   - 未完成或空實作的檔案
echo.
echo ## 恢復檔案
echo.
echo 如需恢復任何檔案，可以從此目錄複製回原位置，或使用 git 歷史記錄。
echo.
echo ## 清理日期
echo.
echo 檔案移至此目錄的日期: %date%
echo.
echo ## Git 備份標籤
echo.
echo 在清理前創建的備份標籤: backup-before-cleanup-*
) > archive\README.md

echo [完成] Archive README 已創建
echo.

REM 步驟 5: 清理 __pycache__
echo [步驟 5] 清理 __pycache__ 目錄
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc >nul 2>&1
echo [完成] __pycache__ 已清理
echo.

REM 步驟 6: 顯示摘要
echo =========================================
echo [完成] 清理完成！
echo =========================================
echo.
echo 已移動的檔案：
echo   - UI 備份: 1 個檔案
echo   - 佔位符: 1 個檔案
echo   - 資料腳本: ~4 個檔案
echo   - 測試檔案: ~20 個檔案
echo   - 評估腳本: 1 個檔案
echo.
echo [重要] 保留的目錄：
echo   - app\core\fmri_processing\pipelines\ (仍在使用中)
echo.
echo 下一步：
echo   1. 執行測試確認系統正常:
echo      python -m pytest tests\test_*.py
echo.
echo   2. 測試 Streamlit 應用:
echo      streamlit run app.py
echo.
echo   3. 如果一切正常，提交變更:
echo      git add -A
echo      git commit -m "Clean up unused code - moved to archive/ (corrected)"
echo.
echo   4. 如需恢復，使用備份標籤:
echo      git tag -l backup-before-cleanup-*
echo      git checkout ^<tag-name^>
echo.
echo [注意] archive\ 目錄已加入 git，可以隨時恢復檔案
echo.
pause
