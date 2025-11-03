# 多模態臨床儀表板設計文件

## 概述

基於現有的 LangGraph 工作流程和 fMRI 處理管道，設計一個完整的多模態臨床儀表板系統。該系統將整合 Web 前端、FastAPI 後端、多模態資料處理、AI 模型推理和臨床報告生成功能，為臨床醫師提供全面的阿茲海默症診斷輔助工具。

## 系統架構

### 整體架構模式
- **前端**: React + TypeScript + Ant Design
- **後端**: FastAPI + Python
- **資料處理**: 現有的 LangGraph 工作流程
- **資料庫**: Neo4j (知識圖譜) + PostgreSQL (患者資料)
- **檔案儲存**: 本地檔案系統 + 雲端儲存
- **AI 推理**: PyTorch + ShuffleNet 模型 + 現有的 GenericInferencePipeline

### 微服務架構
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │  API Gateway    │    │ File Management │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Patient Data    │    │ Workflow Engine │    │ AI Model Hub    │
│ Service         │◄──►│  (LangGraph)    │◄──►│   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Knowledge Graph │    │ Report Generator│    │ Visualization   │
│ Service (Neo4j) │◄──►│   Service       │◄──►│   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 核心組件設計

### 1. 前端架構 (React + TypeScript)

#### 主要頁面結構
```
src/
├── components/
│   ├── common/           # 通用組件
│   ├── patient/          # 患者管理組件
│   ├── analysis/         # 分析相關組件
│   ├── visualization/    # 視覺化組件
│   └── reports/          # 報告組件
├── pages/
│   ├── Dashboard.tsx     # 主儀表板
│   ├── PatientList.tsx   # 患者列表
│   ├── Analysis.tsx      # 分析頁面
│   └── Reports.tsx       # 報告頁面
├── services/
│   ├── api.ts           # API 服務
│   ├── fileUpload.ts    # 檔案上傳服務
│   └── websocket.ts     # 即時通訊
└── types/
    ├── patient.ts       # 患者資料類型
    ├── analysis.ts      # 分析結果類型
    └── api.ts          # API 響應類型
```

#### 關鍵組件設計
- **PatientUploadComponent**: 支援 DICOM/NIfTI 拖拽上傳
- **ModelConfigurationComponent**: ShuffleNet 模型配置和參數設定
- **BrainVisualizationComponent**: 3D/2D 腦部視覺化
- **AnalysisProgressComponent**: 即時分析進度顯示
- **ReportViewerComponent**: 互動式報告檢視器###
 2. 後端 API 架構 (FastAPI)

#### API 路由設計
```python
# app/api/
├── routes/
│   ├── patients.py      # 患者管理 API
│   ├── files.py         # 檔案上傳/管理 API
│   ├── analysis.py      # 分析執行 API
│   ├── models.py        # AI 模型管理 API
│   ├── reports.py       # 報告生成 API
│   └── websocket.py     # WebSocket 連接
├── models/
│   ├── patient.py       # 患者資料模型
│   ├── analysis.py      # 分析結果模型
│   └── report.py        # 報告模型
└── services/
    ├── patient_service.py
    ├── analysis_service.py
    └── report_service.py
```

#### 核心 API 端點
```
POST   /api/patients                    # 創建患者記錄
GET    /api/patients                    # 獲取患者列表
GET    /api/patients/{id}               # 獲取患者詳情
POST   /api/patients/{id}/files         # 上傳患者檔案
POST   /api/analysis/start              # 開始分析
GET    /api/analysis/{id}/status        # 獲取分析狀態
GET    /api/analysis/{id}/results       # 獲取分析結果
POST   /api/reports/generate            # 生成報告
GET    /api/models                      # 獲取可用模型列表
WS     /ws/analysis/{id}                # 分析進度 WebSocket
```

### 3. 資料處理管道整合

#### 現有 LangGraph 工作流程擴展
```python
# 擴展現有的 workflow.py
workflow.add_node("file_validation", validate_uploaded_files)
workflow.add_node("dicom_conversion", convert_dicom_to_nifti)
workflow.add_node("metadata_extraction", extract_clinical_metadata)
workflow.add_node("quality_control", perform_quality_checks)
workflow.add_node("shufflenet_inference", run_shufflenet_model)
workflow.add_node("atlas_integration", integrate_brain_atlases)
workflow.add_node("network_analysis", analyze_functional_networks)
workflow.add_node("clinical_correlation", correlate_with_clinical_data)
```

#### 多模態資料處理流程
```
DICOM/NIfTI Upload → File Validation → Format Conversion
                                            ↓
Clinical Report ← Report Generation ← Metadata Extraction
                                            ↓
                    ↑                Quality Control
Network Analysis ← Atlas Integration ← ShuffleNet Inference
```

### 4. 資料模型設計

#### 患者資料模型
```python
class Patient(BaseModel):
    id: str
    name: str
    age: int
    gender: str
    diagnosis: Optional[str]
    scan_date: datetime
    hospital_info: HospitalInfo
    clinical_notes: Optional[str]
    created_at: datetime
    updated_at: datetime

class HospitalInfo(BaseModel):
    institution_name: str
    department: str
    scanner_model: str
    magnetic_field_strength: float
```

#### 分析結果模型
```python
class AnalysisResult(BaseModel):
    id: str
    patient_id: str
    shufflenet_result: ModelResult
    brain_regions: List[BrainRegion]
    functional_networks: List[FunctionalNetwork]
    quality_metrics: QualityMetrics
    status: AnalysisStatus
    created_at: datetime

class ModelResult(BaseModel):
    model_name: str
    prediction: str
    confidence: float
    grad_cam_paths: List[str]
    processing_time: float

class BrainRegion(BaseModel):
    aal3_label: str
    region_name: str
    activation_score: float
    yeo_network: str
    hemisphere: str
    coordinates: Tuple[float, float, float]
```#
## 5. 視覺化引擎設計

#### 3D 腦部視覺化
- **技術棧**: Three.js + React Three Fiber
- **功能**: 
  - MNI 標準腦模板載入
  - 活化熱圖疊加顯示
  - 互動式腦區選擇
  - 多視角切換（軸狀面、冠狀面、矢狀面）

#### 2D 切片視覺化
- **技術棧**: Canvas API + D3.js
- **功能**:
  - 動態切片瀏覽
  - Grad-CAM 熱圖疊加
  - 腦區邊界顯示
  - 縮放和平移操作

#### 統計圖表
- **技術棧**: Chart.js + React Chart.js 2
- **圖表類型**:
  - 腦區活化強度柱狀圖
  - 功能網路雷達圖
  - 信心分數分布圖
  - 時間序列趨勢圖

### 6. 檔案管理系統

#### 檔案儲存架構
```
storage/
├── patients/
│   └── {patient_id}/
│       ├── raw/              # 原始 DICOM/NIfTI 檔案
│       ├── processed/        # 預處理後的檔案
│       ├── analysis/         # 分析結果檔案
│       │   ├── models/       # 各模型輸出
│       │   ├── heatmaps/     # 熱圖檔案
│       │   └── visualizations/ # 視覺化圖片
│       └── reports/          # 生成的報告
├── templates/                # 腦模板檔案
├── atlases/                  # 腦圖譜檔案
└── temp/                     # 臨時檔案
```

#### 檔案處理服務
```python
class FileManager:
    def upload_patient_files(self, patient_id: str, files: List[UploadFile])
    def convert_dicom_to_nifti(self, dicom_path: str) -> str
    def extract_metadata(self, file_path: str) -> Dict
    def validate_file_format(self, file_path: str) -> bool
    def cleanup_temp_files(self, older_than: timedelta)
```

## 組件間介面設計

### 1. API 介面規範

#### 分析請求介面
```python
class AnalysisRequest(BaseModel):
    patient_id: str
    use_shufflenet: bool = True
    analysis_options: AnalysisOptions
    priority: int = 1

class AnalysisOptions(BaseModel):
    include_grad_cam: bool = True
    include_network_analysis: bool = True
    atlas_type: str = "aal3"
    network_type: str = "yeo7"
    quality_threshold: float = 0.8
```

#### 分析響應介面
```python
class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    progress: float
    estimated_completion: Optional[datetime]
    current_step: str
    results: Optional[AnalysisResult]
```

### 2. WebSocket 事件規範

#### 分析進度事件
```json
{
  "event": "analysis_progress",
  "data": {
    "analysis_id": "uuid",
    "progress": 0.65,
    "current_step": "Running Model Inference",
    "step_details": "Processing with ShuffleNet model",
    "estimated_remaining": "2 minutes"
  }
}
```

#### 錯誤事件
```json
{
  "event": "analysis_error",
  "data": {
    "analysis_id": "uuid",
    "error_type": "FileFormatError",
    "message": "Invalid NIfTI file format",
    "suggestions": ["Check file integrity", "Re-upload file"]
  }
}
```

### 3. 資料庫設計

#### PostgreSQL 表結構
```sql
-- 患者表
CREATE TABLE patients (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    diagnosis VARCHAR(100),
    scan_date TIMESTAMP,
    hospital_info JSONB,
    clinical_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 分析表
CREATE TABLE analyses (
    id UUID PRIMARY KEY,
    patient_id UUID REFERENCES patients(id),
    status VARCHAR(50) NOT NULL,
    model_name VARCHAR(50) DEFAULT 'shufflenet',
    results JSONB,
    quality_metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- 檔案表
CREATE TABLE patient_files (
    id UUID PRIMARY KEY,
    patient_id UUID REFERENCES patients(id),
    file_type VARCHAR(50),
    file_path VARCHAR(500),
    file_size BIGINT,
    metadata JSONB,
    uploaded_at TIMESTAMP DEFAULT NOW()
);
```###
# Neo4j 知識圖譜結構
```cypher
// 腦區節點
CREATE (region:BrainRegion {
    aal3_id: "1",
    name: "Precentral_L",
    hemisphere: "Left",
    coordinates: [x, y, z]
})

// 功能網路節點
CREATE (network:FunctionalNetwork {
    yeo_id: "1",
    name: "Visual",
    description: "Primary visual processing network"
})

// 疾病節點
CREATE (disease:Disease {
    name: "Alzheimer's Disease",
    icd_code: "G30"
})

// 關係
CREATE (region)-[:BELONGS_TO]->(network)
CREATE (region)-[:ASSOCIATED_WITH {strength: 0.8}]->(disease)
```

## 錯誤處理策略

### 1. 檔案上傳錯誤
- **格式驗證**: 檢查 DICOM/NIfTI 檔案完整性
- **大小限制**: 設定合理的檔案大小上限
- **病毒掃描**: 整合防毒軟體檢查
- **重複檢測**: 避免重複上傳相同檔案

### 2. 分析處理錯誤
- **資源不足**: 監控 GPU/CPU 使用率，排隊處理
- **模型載入失敗**: 提供模型健康檢查和自動重啟
- **記憶體溢出**: 實施批次處理和記憶體管理
- **網路中斷**: 支援分析任務的暫停和恢復

### 3. 資料一致性
- **交易管理**: 使用資料庫交易確保資料一致性
- **備份策略**: 定期備份患者資料和分析結果
- **版本控制**: 追蹤分析結果的版本變更
- **審計日誌**: 記錄所有重要操作的日誌

## 測試策略

### 1. 單元測試
- **API 端點測試**: 使用 pytest 測試所有 API 功能
- **資料模型測試**: 驗證資料模型的序列化和驗證
- **檔案處理測試**: 測試各種檔案格式的處理邏輯
- **分析管道測試**: 測試 LangGraph 工作流程的各個節點

### 2. 整合測試
- **端到端測試**: 使用 Playwright 測試完整的使用者流程
- **API 整合測試**: 測試前後端 API 整合
- **資料庫整合測試**: 測試資料庫操作和查詢
- **檔案系統測試**: 測試檔案上傳和儲存功能

### 3. 效能測試
- **負載測試**: 模擬多用戶同時使用系統
- **壓力測試**: 測試系統在高負載下的表現
- **記憶體測試**: 監控記憶體使用和洩漏
- **響應時間測試**: 確保 API 響應時間在可接受範圍內

## 部署架構

### 1. 開發環境
- **Docker Compose**: 本地開發環境容器化
- **熱重載**: 前後端代碼變更自動重載
- **測試資料**: 提供標準測試資料集
- **除錯工具**: 整合開發者工具和除錯器

### 2. 生產環境
- **Kubernetes**: 容器編排和自動擴展
- **負載均衡**: Nginx 反向代理和負載分散
- **監控系統**: Prometheus + Grafana 監控
- **日誌管理**: ELK Stack 集中式日誌管理

### 3. 安全考量
- **身份驗證**: JWT Token 基礎的使用者認證
- **授權控制**: RBAC 角色基礎的存取控制
- **資料加密**: 敏感資料的傳輸和儲存加密
- **HIPAA 合規**: 符合醫療資料隱私法規要求

## 擴展性設計

### 1. 水平擴展
- **微服務架構**: 各服務可獨立擴展
- **資料庫分片**: 支援資料庫水平分割
- **快取策略**: Redis 快取熱點資料
- **CDN 整合**: 靜態資源內容分發網路

### 2. 功能擴展
- **插件系統**: 支援第三方模型和分析工具
- **API 版本控制**: 向後相容的 API 版本管理
- **配置管理**: 動態配置系統參數
- **多租戶支援**: 支援多醫院/機構使用

這個設計提供了一個完整的多模態臨床儀表板架構，整合了你們現有的技術棧和工作流程，同時支援未來的擴展需求。