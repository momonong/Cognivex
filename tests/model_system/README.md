# 通用模型系統測試

這個資料夾包含了通用模型推理系統的測試腳本。

## 測試腳本

### 🏃‍♂️ 快速運行所有測試
```bash
poetry run python tests/model_system/run_all_tests.py
```

### 📋 個別測試腳本

#### 1. `analyze_layers.py`
- **用途**: 分析層選擇和視覺化建議
- **功能**: 
  - 分析 CapsNet 和 MCADNNet 的架構
  - 提供層選擇建議
  - 解釋視覺化效果預期

```bash
poetry run python tests/model_system/analyze_layers.py
```

#### 2. `test_layer_selection.py` 
- **用途**: 詳細的層選擇測試
- **功能**:
  - 測試模型檢查功能
  - 比較不同選層策略
  - 驗證層存在性

```bash
poetry run python tests/model_system/test_layer_selection.py
```

#### 3. `test_improved_selection.py`
- **用途**: 改進策略測試  
- **功能**:
  - 比較原始 vs 改進的選層策略
  - 測試實際流水線集成
  - 提供視覺化建議

```bash
poetry run python tests/model_system/test_improved_selection.py
```

## 測試結果解讀

### ✅ 成功指標
- 所有模型能成功載入
- 層選擇策略運行無錯誤
- 選中的層通過驗證
- 改進策略選擇更合理的層

### 📊 關鍵輸出
- **層選擇結果**: 顯示每個策略選中的層
- **改進對比**: 原始 vs 改進策略的差異
- **視覺化建議**: 每層預期的視覺化效果

## 故障排除

### 常見問題

**Q: MCADNNet 測試失敗？**
A: 通常是設備兼容性問題，檢查 CUDA/MPS 設置

**Q: 層選擇策略沒有改變？**
A: 確保模型適配器使用了正確的策略名稱

**Q: 導入錯誤？**
A: 確保從項目根目錄運行測試

## 開發指南

### 添加新測試
1. 在此資料夾創建新的測試腳本
2. 更新 `run_all_tests.py` 中的測試列表
3. 遵循現有的命名和結構慣例

### 測試新模型
1. 創建新的模型適配器
2. 添加到測試腳本中
3. 定義適當的選層策略

## 相關文件

- `../../app/core/fmri_processing/model_config.py` - 模型配置系統
- `../../app/core/fmri_processing/generic_pipeline_steps.py` - 通用流水線
- `../../app/core/fmri_processing/pipelines/choose_layer.py` - 選層策略
- `../../MIGRATION_GUIDE.md` - 遷移指南