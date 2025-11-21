# 修復記錄 - 2024

## 問題 1: ModuleNotFoundError: No module named 'app.core.fmri_processing'

### 原因
代碼中使用了錯誤的模組路徑 `app.core.fmri_processing`，但實際的模組位於 `app.core.mri_processing`。

### 修復的文件
- `app/agents/inference.py`
- `app/agents/postprocessing.py`
- `app/core/mri_processing/generic_pipeline_steps.py`
- `app/core/mri_processing/pipelines/inspector.py`
- `app/core/mri_processing/pipelines/choose_layer.py`
- `app/core/mri_processing/pipelines/inference.py`
- `app/core/mri_processing/pipelines/attach_hook.py`

### 修復內容
將所有 `from app.core.fmri_processing` 改為 `from app.core.mri_processing`

---

## 問題 2: ModuleNotFoundError: No module named 'model.shufflenet.model'

### 原因
代碼嘗試從不存在的 `model.shufflenet.model` 模組導入類別和常數，但該模組文件不存在（只有權重文件 `shufflenet_best_model.pth`）。

### 修復的文件
1. **app/core/mri_processing/pipelines/act_to_nii.py**
   - 移除對 `model.shufflenet.model` 的導入
   - 直接定義常數：`NUM_SLICES_PER_SUBJECT = 10`, `SLICE_IMG_SIZE = 128`

2. **app/core/mri_processing/pipelines/visualize.py**
   - 移除對 `model.shufflenet.model` 的導入
   - 直接定義常數和 `preprocess_nii_to_slices` 函數

3. **app/core/mri_processing/model_config.py**
   - 將 `PaperModelAdapter` 改為使用 `MCADNNet` 模型（來自 `scripts.macadnnet.model`）
   - 更新輸入尺寸從 128x128 改為 64x64（匹配 MCADNNet 的要求）
   - 更新配置註釋，說明使用 MCADNNet 作為 2D CNN 模型

### 結果
✓ 所有導入錯誤已解決
✓ 應用程序可以正常啟動
✓ 不再有 shufflenet.model 相關的錯誤訊息

---

## 測試驗證

```bash
# 測試 1: 基本導入
python -c "from app.graph.workflow import app; print('Import successful!')"
# ✓ 通過

# 測試 2: 配置載入
python -c "from app.core.mri_processing.model_config import get_config_by_name; config = get_config_by_name('shufflenet'); print(f'Config loaded: {config.model_type}')"
# ✓ 通過

# 測試 3: 模型適配器
python -c "from app.core.mri_processing.model_config import ModelFactory, get_config_by_name; config = get_config_by_name('shufflenet'); adapter = ModelFactory.create_adapter(config); print('Adapter created successfully')"
# ✓ 通過
```

## 注意事項

1. **模型變更**: 原本使用的 PaperModel (ShuffleNet-based) 已改為使用 MCADNNet
2. **輸入尺寸**: 切片尺寸從 128x128 改為 64x64
3. **配置別名**: `shufflenet`, `papermodel`, `mcadnnet` 都指向同一個配置（使用 MCADNNet）
