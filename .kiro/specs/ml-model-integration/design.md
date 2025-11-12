# Design Document

## Overview

本設計文件描述如何將基於 Random Forest 的結構性 MRI 分析模型整合到現有的 LangGraph agent 架構中。整合策略採用模組化設計，確保與現有 fMRI 分析流程共存，並支援未來的多模態分析擴展。

### 核心設計原則

1. **模組化隔離**: 結構性 MRI 分析作為獨立模組，不影響現有 fMRI 流程
2. **統一介面**: 遵循現有 agent 節點的輸入輸出規範（AgentState）
3. **可擴展性**: 設計支援未來添加更多模型或分析模式
4. **效能優化**: 使用快取機制避免重複載入模型和 atlas
5. **錯誤容錯**: 完善的錯誤處理，確保單一模組失敗不影響整體系統

## Architecture

### 系統架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Mode Selector│  │ Model Selector│  │ File Upload  │      │
│  │ (Struct/Func)│  │ (ML/DL Models)│  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph Workflow Layer                    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Conditional Router Node                 │   │
│  │  (根據 analysis_mode 選擇分支)                      │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                              │                   │
│           ▼                              ▼                   │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ Structural MRI   │         │ Functional MRI   │         │
│  │ Branch (NEW)     │         │ Branch (EXISTING)│         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘


### Structural MRI 分析分支詳細架構

```
START (analysis_mode="structural")
  │
  ▼
┌─────────────────────────────────────┐
│ structural_mri_inference            │
│ - 載入 ML 模型                      │
│ - 提取 32 ROI 特徵                  │
│ - 執行預測                          │
│ - 計算信心分數                      │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ structural_feature_analyzer         │
│ - 提取特徵重要性                    │
│ - 識別 Top-N 重要 ROI               │
│ - 準備視覺化數據                    │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ structural_visualizer               │
│ - 生成特徵重要性圖表                │
│ - 生成 3D 腦區視覺化                │
│ - 儲存視覺化結果                    │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ entity_linker (共用)                │
│ - 標準化 ROI 名稱                   │
│ - 連結到知識圖譜                    │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ knowledge_reasoner (共用)           │
│ - 查詢 ROI 臨床意義                 │
│ - 豐富腦區資訊                      │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ report_generator (擴展)             │
│ - 生成結構性 MRI 報告               │
│ - 整合特徵重要性解釋                │
│ - 中英文雙語輸出                    │
└─────────────────────────────────────┘
  │
  ▼
END
```

### 資料流設計

```python
# AgentState 擴展欄位
class AgentState(TypedDict):
    # === 新增欄位 ===
    analysis_mode: Optional[str]  # "structural" or "functional"
    ml_model_type: Optional[str]  # "random_forest", "svm", etc.
    
    # Structural MRI 特定欄位
    roi_features: Optional[Dict[str, float]]  # {roi_name: feature_value}
    feature_importances: Optional[Dict[str, float]]  # {roi_name: importance}
    prediction_confidence: Optional[float]  # 0.0 - 1.0
    
    # 視覺化路徑
    feature_importance_plot_path: Optional[str]
    roi_visualization_path: Optional[str]
```

## Components and Interfaces

### Component 1: ML Model Loader

**職責**: 載入和快取 ML 模型及相關檔案

**檔案位置**: `app/core/ml_processing/model_loader.py`

**介面設計**:
```python
class MLModelLoader:
    """管理 ML 模型的載入和快取"""
    
    def __init__(self, model_dir: str = "model/ml/final"):
        self.model_dir = model_dir
        self._model = None
        self._scaler = None
        self._roi_list = None
        self._feature_names = None
    
    def load_model(self) -> RandomForestClassifier:
        """載入 Random Forest 模型"""
        pass
    
    def load_scaler(self) -> StandardScaler:
        """載入特徵標準化器"""
        pass
    
    def load_roi_list(self) -> List[str]:
        """載入 ROI 列表"""
        pass
    
    def load_feature_names(self) -> List[str]:
        """載入特徵名稱"""
        pass
    
    def get_all_components(self) -> Dict[str, Any]:
        """一次性載入所有組件"""
        pass
```


### Component 2: ROI Feature Extractor

**職責**: 從結構性 MRI 影像提取 32 個 ROI 特徵

**檔案位置**: `app/core/ml_processing/feature_extractor.py`

**介面設計**:
```python
class ROIFeatureExtractor:
    """從 T1 MRI 提取 ROI 特徵"""
    
    def __init__(self, atlas_name: str = "AAL"):
        self.atlas_name = atlas_name
        self._atlas_img = None
        self._masker = None
    
    def load_atlas(self) -> nib.Nifti1Image:
        """載入 AAL atlas"""
        pass
    
    def extract_features(
        self, 
        nii_path: str, 
        roi_list: List[str]
    ) -> np.ndarray:
        """
        提取指定 ROI 的特徵
        
        Args:
            nii_path: T1 MRI 檔案路徑
            roi_list: 要提取的 ROI 名稱列表
        
        Returns:
            shape (32,) 的特徵向量
        """
        pass
    
    def get_roi_mapping(self) -> Dict[str, int]:
        """取得 ROI 名稱到 atlas 索引的映射"""
        pass
```

**實作細節**:
- 使用 `nilearn.input_data.NiftiLabelsMasker`
- 策略: `strategy='mean'` (計算 ROI 內體素的平均值)
- 標準化: 使用載入的 scaler 進行 z-score 標準化

### Component 3: Structural MRI Inference Agent

**職責**: 執行 ML 模型推論的主要 agent 節點

**檔案位置**: `app/agents/structural_mri_inference.py`

**介面設計**:
```python
def run_structural_mri_inference(state: AgentState) -> dict:
    """
    執行結構性 MRI 的 ML 模型推論
    
    Args:
        state: AgentState 包含 fmri_scan_path (實際是 T1 MRI)
    
    Returns:
        更新的 state dict，包含:
        - classification_result: "NC" or "AD"
        - prediction_confidence: float
        - roi_features: Dict[str, float]
        - feature_importances: Dict[str, float]
    """
    pass
```

**處理流程**:
1. 從 state 取得影像路徑
2. 使用 MLModelLoader 載入模型組件
3. 使用 ROIFeatureExtractor 提取特徵
4. 標準化特徵
5. 執行預測
6. 提取特徵重要性
7. 更新 state 並返回

### Component 4: Feature Analyzer Agent

**職責**: 分析特徵重要性並準備視覺化數據

**檔案位置**: `app/agents/structural_feature_analyzer.py`

**介面設計**:
```python
def analyze_feature_importance(state: AgentState) -> dict:
    """
    分析特徵重要性並識別關鍵 ROI
    
    Returns:
        更新的 state dict，包含:
        - activated_regions: List[BrainRegionInfo]
          (按重要性排序的 ROI 資訊)
    """
    pass
```

**處理邏輯**:
1. 從 state 取得 feature_importances
2. 排序並選擇 Top-N (預設 10) 重要特徵
3. 將 ROI 名稱轉換為 BrainRegionInfo 格式
4. 設定 activation_score 為 feature_importance 值
5. 更新 state 的 activated_regions 欄位


### Component 5: Structural Visualizer Agent

**職責**: 生成特徵重要性和腦區視覺化

**檔案位置**: `app/agents/structural_visualizer.py`

**介面設計**:
```python
def generate_structural_visualizations(state: AgentState) -> dict:
    """
    生成結構性 MRI 分析的視覺化
    
    Returns:
        更新的 state dict，包含:
        - visualization_paths: List[str]
          包含特徵重要性圖和腦區視覺化的路徑
    """
    pass

def plot_feature_importance(
    importances: Dict[str, float],
    output_path: str,
    top_n: int = 10
) -> str:
    """生成特徵重要性橫條圖"""
    pass

def plot_roi_on_brain(
    roi_importances: Dict[str, float],
    output_path: str,
    atlas_name: str = "AAL"
) -> str:
    """在 3D 腦模板上標記重要 ROI"""
    pass
```

**視覺化規格**:

1. **特徵重要性圖**:
   - 類型: 水平橫條圖
   - 顯示: Top 10 ROI
   - X 軸: 重要性百分比
   - Y 軸: ROI 名稱
   - 顏色: 漸層色（重要性越高越深）

2. **腦區視覺化**:
   - 使用 `nilearn.plotting.plot_roi`
   - 背景: MNI152 標準腦模板
   - 顏色編碼: 根據重要性
   - 視角: 三視圖（矢狀面、冠狀面、軸向）

### Component 6: Workflow Router

**職責**: 根據分析模式路由到不同的處理分支

**檔案位置**: `app/graph/workflow.py` (修改現有檔案)

**路由邏輯**:
```python
def route_by_analysis_mode(state: AgentState) -> str:
    """
    根據 analysis_mode 決定下一個節點
    
    Returns:
        "structural_branch" or "functional_branch"
    """
    mode = state.get("analysis_mode", "functional")
    if mode == "structural":
        return "structural_mri_inference"
    else:
        return "inference"  # 現有的 fMRI inference
```

**Workflow 更新**:
```python
# 新增節點
workflow.add_node("router", route_by_analysis_mode)
workflow.add_node("structural_mri_inference", run_structural_mri_inference)
workflow.add_node("structural_feature_analyzer", analyze_feature_importance)
workflow.add_node("structural_visualizer", generate_structural_visualizations)

# 更新邊
workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_by_analysis_mode,
    {
        "structural_mri_inference": "structural_mri_inference",
        "inference": "inference"
    }
)

# Structural 分支
workflow.add_edge("structural_mri_inference", "structural_feature_analyzer")
workflow.add_edge("structural_feature_analyzer", "structural_visualizer")
workflow.add_edge("structural_visualizer", "entity_linker")

# Functional 分支 (保持不變)
workflow.add_edge("inference", "filtering")
# ... 其他現有邊
```

### Component 7: Report Generator Extension

**職責**: 擴展現有 report generator 以支援結構性 MRI 報告

**檔案位置**: `app/agents/report_generator.py` (修改現有檔案)

**擴展設計**:
```python
def generate_final_report(state: AgentState) -> dict:
    """擴展以支援雙模態報告生成"""
    
    analysis_mode = state.get("analysis_mode", "functional")
    
    if analysis_mode == "structural":
        return generate_structural_report(state)
    else:
        return generate_functional_report(state)  # 現有邏輯

def generate_structural_report(state: AgentState) -> dict:
    """生成結構性 MRI 分析報告"""
    
    # 收集資訊
    classification = state.get("classification_result")
    confidence = state.get("prediction_confidence")
    top_regions = state.get("activated_regions", [])[:10]
    
    # 建立 prompt
    prompt = f"""
    Generate a clinical report for structural MRI analysis.
    
    Classification: {classification}
    Confidence: {confidence:.2%}
    
    Top 10 Important Brain Regions:
    {format_regions_for_structural_report(top_regions)}
    
    Include:
    1. Primary Assessment (classification and confidence)
    2. Key Structural Findings (important ROIs)
    3. Clinical Interpretation (based on MODEL_OVERALL.md)
    4. Limitations (this is an assistive tool)
    """
    
    # 呼叫 LLM
    report_en = llm_response(prompt, llm_provider="gemini")
    report_zh = translate_to_chinese(report_en)
    
    return {
        "generated_reports": {"en": report_en, "zh": report_zh}
    }
```


## Data Models

### Extended AgentState

```python
from typing import TypedDict, List, Dict, Any, Optional, Literal

class AgentState(TypedDict):
    """擴展的 AgentState 支援雙模態分析"""
    
    # === 原有欄位 (保持不變) ===
    subject_id: str
    fmri_scan_path: str  # 對於結構性 MRI，這會是 T1 影像路徑
    model_path: Optional[str]
    model_name: Optional[str]
    
    # === 新增: 分析模式控制 ===
    analysis_mode: Literal["structural", "functional"]
    ml_model_type: Optional[str]  # "random_forest", "svm", etc.
    
    # === 新增: Structural MRI 特定欄位 ===
    roi_features: Optional[Dict[str, float]]
    # 範例: {"Hippocampus_L": 0.523, "Hippocampus_R": 0.487, ...}
    
    feature_importances: Optional[Dict[str, float]]
    # 範例: {"Cingulum_Post_R": 0.0861, "Lingual_R": 0.0635, ...}
    
    prediction_confidence: Optional[float]
    # 範例: 0.754 (75.4% 信心)
    
    # === 新增: 視覺化路徑 ===
    feature_importance_plot_path: Optional[str]
    roi_visualization_path: Optional[str]
    
    # === 原有欄位繼續 ===
    validated_layers: Optional[List[Dict[str, Any]]]
    final_layers: Optional[List[Dict[str, Any]]]
    # ... 其他現有欄位
```

### BrainRegionInfo Extension

```python
class BrainRegionInfo(TypedDict):
    """擴展以支援結構性 MRI 的 ROI 資訊"""
    
    region_name: str
    activation_score: float  # 對於 ML 模型，這是 feature_importance
    hemisphere: str
    
    # 新增: 結構性 MRI 特定欄位
    feature_value: Optional[float]  # 原始特徵值（標準化後）
    importance_rank: Optional[int]  # 重要性排名
    clinical_relevance: Optional[str]  # 從 MODEL_OVERALL.md 提取的臨床相關性
    
    # 原有欄位
    associated_networks: Optional[List[str]]
    known_functions: Optional[str]
```

### Model Configuration

```python
@dataclass
class MLModelConfig:
    """ML 模型配置"""
    
    model_type: str  # "random_forest"
    model_path: str
    scaler_path: str
    roi_list_path: str
    feature_names_path: str
    
    # ROI 提取配置
    atlas_name: str = "AAL"
    num_features: int = 32
    
    # 視覺化配置
    top_n_features: int = 10
    colormap: str = "RdYlBu_r"
    
    @classmethod
    def from_directory(cls, model_dir: str) -> "MLModelConfig":
        """從模型目錄自動建立配置"""
        return cls(
            model_type="random_forest",
            model_path=f"{model_dir}/final_model.pkl",
            scaler_path=f"{model_dir}/final_scaler.pkl",
            roi_list_path=f"{model_dir}/final_roi_list.csv",
            feature_names_path=f"{model_dir}/final_feature_names.txt"
        )
```

## Error Handling

### 錯誤分類與處理策略

```python
class MLIntegrationError(Exception):
    """ML 整合相關錯誤的基礎類別"""
    pass

class ModelLoadError(MLIntegrationError):
    """模型載入失敗"""
    # 處理: 記錄錯誤，禁用結構性 MRI 功能，顯示友善訊息

class FeatureExtractionError(MLIntegrationError):
    """特徵提取失敗"""
    # 處理: 檢查影像格式，提供診斷資訊，建議使用者檢查檔案

class AtlasLoadError(MLIntegrationError):
    """Atlas 載入失敗"""
    # 處理: 嘗試下載 atlas，提供手動安裝指引

class PredictionError(MLIntegrationError):
    """預測過程失敗"""
    # 處理: 記錄詳細錯誤，回滾到安全狀態
```

### 錯誤處理流程

```python
def run_structural_mri_inference(state: AgentState) -> dict:
    """帶完整錯誤處理的推論函式"""
    
    try:
        # 1. 驗證輸入
        if not state.get("fmri_scan_path"):
            raise ValueError("Missing MRI scan path")
        
        # 2. 載入模型 (帶重試機制)
        try:
            model_components = load_model_with_retry(max_retries=3)
        except ModelLoadError as e:
            return {
                "error_log": state.get("error_log", []) + [
                    f"Model load failed: {e}. Structural MRI analysis disabled."
                ],
                "classification_result": "ERROR: Model unavailable"
            }
        
        # 3. 提取特徵 (帶驗證)
        try:
            features = extract_and_validate_features(
                state["fmri_scan_path"],
                expected_shape=(32,)
            )
        except FeatureExtractionError as e:
            return {
                "error_log": state.get("error_log", []) + [
                    f"Feature extraction failed: {e}. "
                    f"Please check if the file is a valid T1 MRI."
                ]
            }
        
        # 4. 執行預測
        prediction, confidence = model_components["model"].predict_proba(features)
        
        # 5. 記錄成功
        trace = f"Structural MRI inference complete: {prediction} ({confidence:.2%})"
        
        return {
            "classification_result": prediction,
            "prediction_confidence": confidence,
            "trace_log": state.get("trace_log", []) + [trace]
        }
        
    except Exception as e:
        # 捕獲所有未預期的錯誤
        error_msg = f"Unexpected error in structural MRI inference: {type(e).__name__}: {e}"
        return {
            "error_log": state.get("error_log", []) + [error_msg],
            "classification_result": "ERROR"
        }
```


## Testing Strategy

### Unit Testing

**測試範圍**:
1. MLModelLoader - 模型載入功能
2. ROIFeatureExtractor - 特徵提取正確性
3. 各個 agent 節點的輸入輸出

**測試檔案結構**:
```
tests/
├── test_ml_model_loader.py
├── test_roi_feature_extractor.py
├── test_structural_agents.py
└── fixtures/
    ├── mock_t1_mri.nii.gz
    ├── mock_model.pkl
    └── mock_scaler.pkl
```

**關鍵測試案例**:

```python
# test_ml_model_loader.py
def test_load_model_success():
    """測試成功載入模型"""
    loader = MLModelLoader("model/ml/final")
    model = loader.load_model()
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 500

def test_load_model_missing_file():
    """測試檔案缺失時的錯誤處理"""
    loader = MLModelLoader("nonexistent/path")
    with pytest.raises(ModelLoadError):
        loader.load_model()

# test_roi_feature_extractor.py
def test_extract_features_correct_shape():
    """測試特徵提取輸出形狀正確"""
    extractor = ROIFeatureExtractor()
    features = extractor.extract_features(
        "tests/fixtures/mock_t1_mri.nii.gz",
        roi_list=["Hippocampus_L", "Hippocampus_R"]
    )
    assert features.shape == (2,)

def test_extract_features_with_invalid_roi():
    """測試無效 ROI 名稱的處理"""
    extractor = ROIFeatureExtractor()
    with pytest.raises(ValueError, match="Invalid ROI"):
        extractor.extract_features(
            "tests/fixtures/mock_t1_mri.nii.gz",
            roi_list=["NonexistentROI"]
        )
```

### Integration Testing

**測試完整 workflow**:

```python
# test_structural_workflow_integration.py
def test_full_structural_pipeline():
    """測試完整的結構性 MRI 分析流程"""
    
    initial_state = {
        "subject_id": "test_sub_01",
        "fmri_scan_path": "tests/fixtures/mock_t1_mri.nii.gz",
        "analysis_mode": "structural",
        "trace_log": [],
        "error_log": []
    }
    
    # 執行 workflow
    final_state = app.invoke(initial_state)
    
    # 驗證輸出
    assert "classification_result" in final_state
    assert final_state["classification_result"] in ["NC", "AD"]
    assert "prediction_confidence" in final_state
    assert 0 <= final_state["prediction_confidence"] <= 1
    assert "feature_importances" in final_state
    assert len(final_state["feature_importances"]) == 32
    assert "visualization_paths" in final_state
    assert len(final_state["error_log"]) == 0

def test_workflow_routing():
    """測試 workflow 正確路由到結構性分支"""
    
    state_structural = {"analysis_mode": "structural"}
    state_functional = {"analysis_mode": "functional"}
    
    assert route_by_analysis_mode(state_structural) == "structural_mri_inference"
    assert route_by_analysis_mode(state_functional) == "inference"
```

### End-to-End Testing

**使用真實數據測試**:

```python
# test_e2e_structural_analysis.py
@pytest.mark.slow
@pytest.mark.requires_real_data
def test_real_subject_analysis():
    """使用真實受試者數據進行端到端測試"""
    
    # 使用一個已知的測試受試者
    test_subject = "sub-ADNI002S0295"
    t1_path = f"data/processed/structural/{test_subject}/T1.nii.gz"
    
    if not os.path.exists(t1_path):
        pytest.skip("Real data not available")
    
    initial_state = {
        "subject_id": test_subject,
        "fmri_scan_path": t1_path,
        "analysis_mode": "structural"
    }
    
    final_state = app.invoke(initial_state)
    
    # 驗證結果合理性
    assert final_state["classification_result"] in ["NC", "AD"]
    assert final_state["prediction_confidence"] > 0.5  # 合理的信心分數
    
    # 驗證視覺化檔案存在
    for viz_path in final_state["visualization_paths"]:
        assert os.path.exists(viz_path)
    
    # 驗證報告生成
    assert "generated_reports" in final_state
    assert "en" in final_state["generated_reports"]
    assert "zh" in final_state["generated_reports"]
```

### Performance Testing

**效能基準測試**:

```python
# test_performance.py
def test_inference_speed():
    """測試推論速度符合需求 (< 5 秒)"""
    
    import time
    
    initial_state = {
        "subject_id": "perf_test",
        "fmri_scan_path": "tests/fixtures/mock_t1_mri.nii.gz",
        "analysis_mode": "structural"
    }
    
    start_time = time.time()
    final_state = app.invoke(initial_state)
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 5.0, f"Inference took {elapsed_time:.2f}s, exceeds 5s limit"

def test_model_caching():
    """測試模型快取有效性"""
    
    loader = MLModelLoader()
    
    # 第一次載入
    start = time.time()
    model1 = loader.load_model()
    first_load_time = time.time() - start
    
    # 第二次載入 (應該從快取)
    start = time.time()
    model2 = loader.load_model()
    cached_load_time = time.time() - start
    
    assert cached_load_time < first_load_time * 0.1  # 快取應該快 10 倍以上
    assert model1 is model2  # 應該是同一個物件
```

## UI/UX Considerations

### Streamlit UI 更新設計

**側邊欄新增控制項**:

```python
# app.py 更新
st.sidebar.header("Analysis Configuration")

# 1. 分析模式選擇
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    options=["Functional MRI (fMRI)", "Structural MRI (T1)"],
    help="Select the type of MRI analysis to perform"
)

# 2. 根據模式顯示對應的模型選擇
if analysis_mode == "Structural MRI (T1)":
    st.sidebar.info("Using Random Forest ML Model (32 ROIs)")
    model_info = {
        "Type": "Random Forest Classifier",
        "Features": "32 AAL ROIs",
        "Accuracy": "75.4% (CV)",
        "Training Data": "65 subjects (ADNI)"
    }
    for key, value in model_info.items():
        st.sidebar.caption(f"**{key}:** {value}")
else:
    # 現有的深度學習模型選擇
    selected_model = st.sidebar.selectbox(
        "Select DL Model",
        options=["ShuffleNet", "CapsNet", "MCADNNet"]
    )
```

**結果顯示區域**:

```python
# 結構性 MRI 結果顯示
if st.session_state.get("analysis_mode") == "structural":
    st.header("Structural MRI Analysis Results")
    
    # 1. 預測結果卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Classification",
            final_state["classification_result"],
            delta="High Confidence" if final_state["prediction_confidence"] > 0.8 else None
        )
    with col2:
        st.metric(
            "Confidence Score",
            f"{final_state['prediction_confidence']:.1%}"
        )
    with col3:
        st.metric(
            "Model Type",
            "Random Forest"
        )
    
    # 2. 特徵重要性視覺化
    st.subheader("Feature Importance Analysis")
    st.image(
        final_state["feature_importance_plot_path"],
        caption="Top 10 Most Important Brain Regions"
    )
    
    # 3. 3D 腦區視覺化
    st.subheader("Brain Region Visualization")
    st.image(
        final_state["roi_visualization_path"],
        caption="Important ROIs Highlighted on Standard Brain Template"
    )
    
    # 4. 詳細 ROI 資訊表格
    st.subheader("Detailed ROI Information")
    roi_df = pd.DataFrame(final_state["activated_regions"])
    st.dataframe(
        roi_df[["region_name", "activation_score", "clinical_relevance"]],
        use_container_width=True
    )
```

### 使用者體驗優化

**進度指示**:
```python
# 分階段顯示進度
progress_bar = st.progress(0)
status_text = st.empty()

status_text.text("Loading ML model...")
progress_bar.progress(20)

status_text.text("Extracting ROI features...")
progress_bar.progress(50)

status_text.text("Running prediction...")
progress_bar.progress(80)

status_text.text("Generating visualizations...")
progress_bar.progress(100)
```

**錯誤訊息友善化**:
```python
# 將技術錯誤轉換為使用者友善訊息
ERROR_MESSAGES = {
    "ModelLoadError": "⚠️ Unable to load the analysis model. Please contact support.",
    "FeatureExtractionError": "⚠️ Could not process the MRI image. Please ensure it's a valid T1-weighted scan.",
    "AtlasLoadError": "⚠️ Brain atlas not found. Attempting to download...",
}

def display_user_friendly_error(error: Exception):
    error_type = type(error).__name__
    friendly_msg = ERROR_MESSAGES.get(error_type, "An unexpected error occurred.")
    st.error(friendly_msg)
    
    with st.expander("Technical Details"):
        st.code(str(error))
```


## Performance Optimization

### Caching Strategy

**模型快取**:
```python
# 使用 Streamlit 的快取裝飾器
@st.cache_resource
def load_ml_model_cached():
    """快取 ML 模型，避免重複載入"""
    loader = MLModelLoader()
    return loader.get_all_components()

# 使用
model_components = load_ml_model_cached()
```

**Atlas 快取**:
```python
@st.cache_data
def load_aal_atlas_cached():
    """快取 AAL atlas"""
    from nilearn import datasets
    return datasets.fetch_atlas_aal()

# 使用
atlas = load_aal_atlas_cached()
```

**特徵提取優化**:
```python
class ROIFeatureExtractor:
    def __init__(self):
        self._masker = None  # 快取 masker 物件
    
    def extract_features(self, nii_path, roi_list):
        # 只在第一次建立 masker
        if self._masker is None:
            self._masker = NiftiLabelsMasker(
                labels_img=self.atlas_img,
                standardize=False
            )
            self._masker.fit()  # 預先 fit
        
        # 後續呼叫直接使用快取的 masker
        return self._masker.transform(nii_path)
```

### Memory Management

**大型物件管理**:
```python
import gc

def run_structural_mri_inference(state: AgentState) -> dict:
    """帶記憶體管理的推論"""
    
    try:
        # 載入影像
        img = nib.load(state["fmri_scan_path"])
        
        # 提取特徵
        features = extract_features(img)
        
        # 釋放影像記憶體
        del img
        gc.collect()
        
        # 執行預測
        prediction = model.predict(features)
        
        return {"classification_result": prediction}
        
    finally:
        # 確保清理
        gc.collect()
```

### Parallel Processing

**批次處理支援** (未來擴展):
```python
from concurrent.futures import ThreadPoolExecutor

def batch_structural_analysis(subject_list: List[str]) -> List[dict]:
    """批次處理多個受試者"""
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(analyze_single_subject, subject_id)
            for subject_id in subject_list
        ]
        
        results = [future.result() for future in futures]
    
    return results
```

## Security Considerations

### 檔案路徑驗證

```python
import os
from pathlib import Path

def validate_file_path(file_path: str, allowed_extensions: List[str]) -> bool:
    """驗證檔案路徑安全性"""
    
    # 1. 檢查路徑遍歷攻擊
    abs_path = os.path.abspath(file_path)
    if not abs_path.startswith(os.path.abspath("data/")):
        raise SecurityError("File path outside allowed directory")
    
    # 2. 檢查副檔名
    if not any(file_path.endswith(ext) for ext in allowed_extensions):
        raise SecurityError(f"Invalid file extension. Allowed: {allowed_extensions}")
    
    # 3. 檢查檔案存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return True

# 使用
try:
    validate_file_path(
        state["fmri_scan_path"],
        allowed_extensions=[".nii", ".nii.gz"]
    )
except SecurityError as e:
    return {"error_log": [str(e)]}
```

### 模型完整性驗證

```python
import hashlib

def verify_model_integrity(model_path: str, expected_hash: str) -> bool:
    """驗證模型檔案完整性"""
    
    with open(model_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    if file_hash != expected_hash:
        raise SecurityError("Model file integrity check failed")
    
    return True

# 配置檔案中儲存預期的 hash
MODEL_HASHES = {
    "model/ml/final/final_model.pkl": "abc123...",
    "model/ml/final/final_scaler.pkl": "def456..."
}
```

### 輸入驗證

```python
def validate_state_input(state: AgentState) -> None:
    """驗證 state 輸入的完整性和安全性"""
    
    # 1. 必要欄位檢查
    required_fields = ["subject_id", "fmri_scan_path", "analysis_mode"]
    for field in required_fields:
        if field not in state:
            raise ValueError(f"Missing required field: {field}")
    
    # 2. 欄位類型檢查
    if not isinstance(state["subject_id"], str):
        raise TypeError("subject_id must be string")
    
    # 3. 值範圍檢查
    if state["analysis_mode"] not in ["structural", "functional"]:
        raise ValueError("Invalid analysis_mode")
    
    # 4. Subject ID 格式驗證 (防止注入攻擊)
    import re
    if not re.match(r"^[a-zA-Z0-9_-]+$", state["subject_id"]):
        raise ValueError("Invalid subject_id format")
```

## Deployment Considerations

### 環境需求

**Python 套件**:
```txt
# requirements.txt 新增
scikit-learn==1.3.0
nilearn==0.10.1
nibabel==5.1.0
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
```

**系統需求**:
- Python 3.9+
- 記憶體: 最少 8GB (建議 16GB)
- 儲存空間: 5GB (包含 atlas 和模型)

### 配置管理

**config.yaml**:
```yaml
ml_model:
  model_dir: "model/ml/final"
  atlas_name: "AAL"
  cache_enabled: true
  
  feature_extraction:
    strategy: "mean"
    standardize: false
  
  visualization:
    top_n_features: 10
    colormap: "RdYlBu_r"
    dpi: 300
  
  performance:
    max_cache_size: 1000  # MB
    enable_parallel: false
```

### 監控與日誌

**結構化日誌**:
```python
import logging
import json

logger = logging.getLogger("ml_integration")

def log_inference_event(
    subject_id: str,
    prediction: str,
    confidence: float,
    duration: float
):
    """記錄推論事件"""
    
    log_data = {
        "event": "structural_mri_inference",
        "subject_id": subject_id,
        "prediction": prediction,
        "confidence": confidence,
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(json.dumps(log_data))
```

**效能監控**:
```python
from functools import wraps
import time

def monitor_performance(func):
    """監控函式執行時間"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        logger.info(f"{func.__name__} took {duration:.2f}s")
        
        # 如果超過閾值，發出警告
        if duration > 10.0:
            logger.warning(f"{func.__name__} exceeded 10s threshold")
        
        return result
    
    return wrapper

@monitor_performance
def run_structural_mri_inference(state: AgentState) -> dict:
    # ... 實作
    pass
```

## Migration Plan

### Phase 1: 基礎整合 (Week 1-2)
1. 建立 `app/core/ml_processing/` 模組
2. 實作 MLModelLoader 和 ROIFeatureExtractor
3. 建立基本的 unit tests
4. 驗證模型載入和特徵提取功能

### Phase 2: Agent 節點開發 (Week 3-4)
1. 實作 structural_mri_inference agent
2. 實作 structural_feature_analyzer agent
3. 實作 structural_visualizer agent
4. 整合測試三個 agent 的協作

### Phase 3: Workflow 整合 (Week 5)
1. 更新 workflow.py 加入路由邏輯
2. 擴展 AgentState 定義
3. 整合現有的 entity_linker 和 knowledge_reasoner
4. 端到端測試完整流程

### Phase 4: UI 整合 (Week 6)
1. 更新 Streamlit UI 加入模式選擇
2. 實作結果顯示頁面
3. 加入進度指示和錯誤處理
4. 使用者體驗測試

### Phase 5: 報告生成 (Week 7)
1. 擴展 report_generator 支援結構性 MRI
2. 整合 MODEL_OVERALL.md 的臨床知識
3. 測試中英文報告生成
4. 臨床專家審查報告內容

### Phase 6: 優化與部署 (Week 8)
1. 效能優化和快取實作
2. 完整的測試覆蓋
3. 文件撰寫
4. 部署到測試環境

## Future Enhancements

### 短期 (3-6 個月)
1. **多模態融合**: 結合結構性和功能性 MRI 的預測
2. **不確定性量化**: 提供預測的置信區間
3. **批次處理**: 支援一次分析多個受試者
4. **模型版本管理**: 支援多個模型版本切換

### 中期 (6-12 個月)
1. **深度學習模型**: 整合 3D CNN 用於結構性 MRI
2. **縱向分析**: 支援追蹤同一受試者的變化
3. **亞型分類**: 識別 AD 的不同亞型
4. **自動化報告**: 更智能的臨床報告生成

### 長期 (1-2 年)
1. **多中心驗證**: 在不同醫院的數據上驗證
2. **臨床試驗整合**: 支援臨床試驗的數據分析
3. **預後預測**: 預測疾病進展速度
4. **治療建議**: 基於分析結果提供個性化建議
