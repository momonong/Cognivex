# 多模態臨床儀表板 API

基於 FastAPI 的多模態臨床儀表板後端服務，整合 fMRI 分析、AI 模型推理、腦圖譜視覺化和臨床報告生成功能。

## 🚀 快速開始

### 啟動 API 服務

```bash
# 方法 1: 使用啟動腳本
python run_api.py

# 方法 2: 直接使用 uvicorn
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 訪問 API 文檔

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI JSON**: http://localhost:8000/api/openapi.json

## 📋 API 端點概覽

### 🏥 患者管理 (`/api/patients`)

- `POST /api/patients/` - 創建患者記錄
- `GET /api/patients/` - 獲取患者列表
- `GET /api/patients/{id}` - 獲取患者詳情
- `PUT /api/patients/{id}` - 更新患者資訊
- `DELETE /api/patients/{id}` - 刪除患者記錄
- `GET /api/patients/{id}/summary` - 獲取患者摘要

### 📁 檔案管理 (`/api/files`)

- `POST /api/files/upload/{patient_id}` - 上傳患者檔案
- `GET /api/files/patient/{patient_id}` - 獲取患者檔案列表
- `GET /api/files/{file_id}` - 獲取檔案詳情
- `DELETE /api/files/{file_id}` - 刪除檔案
- `POST /api/files/{file_id}/validate` - 驗證檔案完整性

### 🧠 分析服務 (`/api/analysis`)

- `POST /api/analysis/start` - 開始 ShuffleNet 分析
- `GET /api/analysis/{id}/status` - 獲取分析狀態
- `GET /api/analysis/{id}/results` - 獲取分析結果
- `GET /api/analysis/patient/{patient_id}` - 獲取患者分析記錄
- `DELETE /api/analysis/{id}` - 取消/刪除分析

### 🤖 模型管理 (`/api/models`)

- `GET /api/models/` - 獲取可用模型列表
- `GET /api/models/{model_id}` - 獲取模型詳情
- `POST /api/models/{model_id}/health-check` - 模型健康檢查
- `GET /api/models/{model_id}/config` - 獲取模型配置

### 📊 報告生成 (`/api/reports`)

- `POST /api/reports/generate/{analysis_id}` - 生成臨床報告
- `GET /api/reports/{report_id}` - 獲取報告資訊
- `GET /api/reports/{report_id}/content` - 獲取報告內容
- `GET /api/reports/{report_id}/download` - 下載報告檔案
- `DELETE /api/reports/{report_id}` - 刪除報告

### 🔌 WebSocket (`/ws`)

- `WS /ws/analysis/{analysis_id}` - 分析進度即時通知
- `WS /ws/test` - WebSocket 測試端點

## 💡 使用範例

### 創建患者

```bash
curl -X POST "http://localhost:8000/api/patients/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "王小明",
    "age": 65,
    "gender": "M",
    "diagnosis": "AD",
    "scan_date": "2025-01-30T10:00:00Z",
    "hospital_info": {
      "institution_name": "台北榮民總醫院",
      "department": "神經內科",
      "scanner_model": "Siemens Magnetom Prisma",
      "magnetic_field_strength": 3.0
    }
  }'
```

### 上傳 fMRI 檔案

```bash
curl -X POST "http://localhost:8000/api/files/upload/{patient_id}" \
  -F "files=@fmri_scan.nii.gz" \
  -F "description=功能性 MRI 掃描"
```

### 開始分析

```bash
curl -X POST "http://localhost:8000/api/analysis/start" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "patient_123",
    "fmri_file_path": "/storage/patients/patient_123/raw/fmri_scan.nii.gz",
    "analysis_options": {
      "include_grad_cam": true,
      "include_network_analysis": true,
      "atlas_type": "aal3",
      "network_type": "yeo7"
    }
  }'
```

### WebSocket 連接 (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analysis/analysis_123');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('分析進度:', data);
};

ws.onopen = function() {
    console.log('WebSocket 連接已建立');
};
```

## 🧪 測試

### 運行 API 測試

```bash
# 基本功能測試
python test_api.py

# 簡單測試
python simple_test.py
```

### 測試覆蓋範圍

- ✅ 患者管理 CRUD 操作
- ✅ 檔案上傳和驗證
- ✅ 分析工作流程
- ✅ 模型管理和健康檢查
- ✅ 報告生成和下載
- ✅ WebSocket 即時通訊
- ✅ 錯誤處理和驗證

## 🏗️ 架構特點

### 技術棧

- **FastAPI**: 現代、高效能的 Web 框架
- **Pydantic**: 資料驗證和序列化
- **Uvicorn**: ASGI 服務器
- **WebSocket**: 即時通訊支援
- **整合現有**: 與 LangGraph 工作流程整合

### 設計原則

- **模組化**: 清晰的路由和模型分離
- **可擴展**: 支援未來功能擴展
- **標準化**: RESTful API 設計
- **文檔化**: 自動生成 API 文檔
- **測試友好**: 完整的測試覆蓋

### 檔案結構

```
app/api/
├── __init__.py
├── main.py              # FastAPI 主應用程式
├── models/              # Pydantic 資料模型
│   ├── __init__.py
│   ├── patient.py       # 患者模型
│   ├── file.py          # 檔案模型
│   └── analysis.py      # 分析模型
└── routes/              # API 路由
    ├── __init__.py
    ├── patients.py      # 患者管理路由
    ├── files.py         # 檔案管理路由
    ├── analysis.py      # 分析服務路由
    ├── models.py        # 模型管理路由
    ├── reports.py       # 報告生成路由
    └── websocket.py     # WebSocket 路由
```

## 🔧 配置

### 環境變數

- `API_HOST`: 服務主機 (預設: 0.0.0.0)
- `API_PORT`: 服務埠號 (預設: 8000)
- `API_RELOAD`: 熱重載 (預設: true)
- `API_LOG_LEVEL`: 日誌級別 (預設: info)

### 存儲目錄

- `storage/patients/`: 患者檔案存儲
- `storage/temp/`: 臨時檔案
- `storage/templates/`: 腦模板檔案
- `storage/atlases/`: 腦圖譜檔案
- `storage/reports/`: 生成的報告

## 🚧 後續開發

### 已完成 ✅

- [x] FastAPI 基礎架構
- [x] 患者管理 API
- [x] 檔案管理 API
- [x] 分析服務 API (模擬)
- [x] 模型管理 API
- [x] 報告生成 API (基礎)
- [x] WebSocket 支援
- [x] API 測試

### 待實作 🔄

- [ ] 資料庫整合 (PostgreSQL + Neo4j)
- [ ] 真實 LangGraph 工作流程整合
- [ ] 腦圖譜服務整合
- [ ] 完整的報告生成 (PDF)
- [ ] 使用者認證和授權
- [ ] 檔案上傳進度追蹤
- [ ] 批次分析功能
- [ ] 系統監控和日誌

## 📞 支援

如有問題或建議，請參考：

- API 文檔: http://localhost:8000/api/docs
- 健康檢查: http://localhost:8000/api/health
- 測試腳本: `python test_api.py`