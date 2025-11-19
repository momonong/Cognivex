# 系統功能介紹 (System Introduction)

## 概述 (Overview)

這是一個基於 AI 的阿茲海默症診斷輔助系統，整合了深度學習、機器學習和知識圖譜技術，提供功能性 fMRI 和結構性 MRI 的多模態分析能力。

**核心價值：**
- 🧠 多模態腦影像分析（fMRI + sMRI）
- 🤖 多種 AI 模型支持（3D CNN、CapsNet、Random Forest）
- 📊 可解釋性分析（GradCAM、SHAP、特徵重要性）
- 🔬 知識圖譜增強（腦區功能、神經網絡關聯）
- 📝 自動化臨床報告生成（中英雙語）
- 🔒 隱私保護模式（本地 LLM 支持）

---

## 系統架構 (System Architecture)

### 1. 核心模組 (`app/`)

系統採用模組化設計，主要分為以下幾個部分：

```
app/
├── agents/          # 工作流節點（推理、分析、報告生成）
├── core/            # 核心處理邏輯
│   ├── fmri_processing/    # fMRI 處理管線
│   ├── ml_processing/      # ML 模型載入與特徵提取
│   ├── knowledge_graph/    # 知識圖譜查詢
│   └── vision/             # 視覺解釋工具
├── graph/           # LangGraph 工作流定義
├── services/        # 外部服務（LLM、Neo4j）
└── ui/              # Streamlit UI 組件
```

---

## 功能模組詳解

### 📊 A. 影像分析模組 (`app/agents/`)

#### A1. 功能性 fMRI 分析
**檔案：** `inference.py`, `filtering.py`, `postprocessing.py`

**功能：**
- 支援多種深度學習模型（CapsNet、ShuffleNet、MCADNNet）
- 自動化層選擇與驗證
- GradCAM 激活圖生成
- 腦區激活分析

**支援模型：**
- **CapsNet3D**: 3D 膠囊網絡，適合時序 fMRI 數據
- **ShuffleNet (PaperModel)**: 2D 切片 CNN，輕量高效
- **MCADNNet**: 多通道注意力網絡

**輸出：**
- 分類結果（AD/NC/MCI）
- 激活腦區列表
- 可視化熱力圖

#### A2. 結構性 MRI 分析
**檔案：** `structural_mri_inference.py`, `cnn_rf_inference.py`

**功能：**
- **傳統 ML 方法**：基於 32 個 AAL ROI 特徵的隨機森林分類器
- **CNN-RF 混合方法**：3D CNN 特徵提取 + 隨機森林分類
- 端到端推理（從原始 MRI 到預測）
- SHAP 局部可解釋性分析

**模型配置：**
```python
# 傳統 ML (32 ROIs)
model: Random Forest
features: 32 AAL ROIs (海馬迴、杏仁核等)
accuracy: 75.4% (5-fold CV)

# CNN-RF (多模態)
model: 3D CNN + Random Forest
features: GM (灰質) + FA (各向異性分數) + MD (平均擴散率)
modalities: T1-weighted + DTI
```

**輸出：**
- 分類結果（NC/AD）
- 預測信心度
- ROI 特徵值
- 特徵重要性排名
- SHAP 值（局部解釋）

---

### 🧪 B. 特徵分析模組

#### B1. 結構特徵分析器
**檔案：** `structural_feature_analyzer.py`

**功能：**
- 從模型中提取特徵重要性
- 將 ROI 特徵轉換為標準化腦區資訊
- 重要性排名與篩選

**輸出格式：**
```python
BrainRegionInfo = {
    "region_name": "Hippocampus_L",
    "activation_score": 0.0856,  # 特徵重要性
    "hemisphere": "Left",
    "feature_value": -0.234,     # 標準化特徵值
    "importance_rank": 1,
    "associated_networks": [...],  # 由知識圖譜填充
    "known_functions": [...]       # 由知識圖譜填充
}
```

#### B2. 視覺化生成器
**檔案：** `structural_visualizer.py`

**功能：**
- 特徵重要性柱狀圖（中英雙語）
- 3D 腦區可視化（基於 MNI152 模板）
- 多視角切片展示（矢狀面、冠狀面、軸向）

**技術棧：**
- `matplotlib` + `seaborn`: 統計圖表
- `nilearn`: 神經影像可視化
- `nibabel`: NIfTI 檔案處理

---

### 🔗 C. 知識圖譜模組 (`app/core/knowledge_graph/`)

#### C1. 實體連結器
**檔案：** `entity_linker.py`

**功能：**
- 將 AI 模型輸出的「髒」腦區名稱標準化
- 使用 LLM 進行智能匹配
- 與 Neo4j 資料庫中的權威名稱對齊

**流程：**
```
AI 輸出: ["Angular_R", "Precuneus_L", "Hippocampus"]
    ↓
LLM 匹配
    ↓
標準化: ["Angular_R 70", "Precuneus_L 71", "Hippocampus_L 41"]
```

#### C2. 知識推理引擎
**檔案：** `query_engine.py`

**功能：**
- 查詢腦區的神經網絡歸屬（Yeo 7-network）
- 查詢腦區的已知功能
- 查詢 AD 相關性

**知識圖譜結構：**
```cypher
(Region)-[:BELONGS_TO]->(YeoNetwork)
(Region)-[:HAS_FUNCTION]->(Function)
(Region {ad_associated: boolean})
```

**範例查詢結果：**
```json
{
  "region": "Hippocampus_L 41",
  "networks": ["Default Mode Network"],
  "functions": ["記憶形成", "空間導航", "情緒調節"],
  "isADAssociated": true
}
```

---

### 🤖 D. LLM 服務模組 (`app/services/llm_providers/`)

#### D1. 多 LLM 支援
**檔案：** `config.py`, `bedrock.py`, `ollama.py`, `gemini.py`

**支援的 LLM：**
1. **AWS Bedrock** (雲端)
   - Claude 3 Haiku (快速經濟)
   - Claude 3 Sonnet (平衡)
   - Claude 3 Opus (最佳品質)

2. **Ollama** (本地 - 隱私保護)
   - Llama 3.2 3B (輕量)
   - Llama 3.1 8B (平衡)
   - Meditron 7B (醫療專用)

3. **Google Gemini** (雲端)
   - Gemini Pro

#### D2. 隱私模式
**環境變數配置：**
```bash
# 啟用隱私模式（使用本地 Ollama）
PRIVACY_MODE=true
LLM_PROVIDER=ollama

# 使用雲端服務
PRIVACY_MODE=false
LLM_PROVIDER=aws_bedrock
```

**優勢：**
- ✅ 病患資料不離開本地環境
- ✅ 符合 HIPAA/GDPR 規範
- ✅ 無需網路連接即可運行

---

### 📝 E. 報告生成模組

#### E1. 影像解釋器
**檔案：** `image_explainer.py`

**功能：**
- 使用視覺 LLM 分析激活圖
- 生成自然語言描述
- 整合分類結果與腦區資訊

#### E2. 報告生成器
**檔案：** `report_generator.py`

**功能：**
- **功能性 MRI 報告**：敘述性文字報告（中英雙語）
- **結構性 MRI 報告**：結構化 JSON 報告（中英雙語）

**結構化報告格式：**
```json
{
  "risk_assessment": {
    "level": "High Risk / Low Risk",
    "confidence": 0.85,
    "primary_finding": "主要發現摘要"
  },
  "key_findings": {
    "structural_changes": [
      {
        "finding": "海馬迴體積減少",
        "severity": "Moderate",
        "significance": "High"
      }
    ],
    "volumetric_analysis": [...]
  },
  "clinical_interpretation": {
    "summary": "臨床摘要",
    "ad_indicators": ["指標1", "指標2"],
    "protective_factors": ["保護因子1"]
  },
  "recommendations": {
    "immediate_actions": ["建議1", "建議2"],
    "monitoring": ["監測項目1"],
    "additional_tests": ["額外檢查1"]
  },
  "limitations": ["限制1", "限制2"]
}
```

---

### 🔄 F. 工作流管理 (`app/graph/`)

#### F1. 狀態管理
**檔案：** `state.py`

**AgentState 結構：**
```python
{
    # 輸入
    "subject_id": str,
    "fmri_scan_path": str,
    "analysis_mode": "structural" | "functional",
    "model_type": "legacy" | "cnn_rf",
    
    # 中間結果
    "validated_layers": [...],
    "activated_regions": [...],
    "clean_region_names": [...],
    
    # 最終輸出
    "classification_result": str,
    "prediction_confidence": float,
    "roi_features": {...},
    "feature_importances": {...},
    "structured_report": {...},
    "visualization_paths": [...],
    
    # 系統追蹤
    "trace_log": [...],
    "error_log": [...]
}
```

#### F2. 工作流定義
**檔案：** `workflow.py`

**工作流分支：**

```
START
  ↓
[路由器] 根據 analysis_mode 分流
  ↓
  ├─→ [功能性 fMRI 分支]
  │     ↓
  │   inference → filtering → post_processing
  │
  ├─→ [結構性 MRI - 傳統 ML]
  │     ↓
  │   structural_mri_inference → feature_analyzer → visualizer
  │
  └─→ [結構性 MRI - CNN-RF]
        ↓
      cnn_rf_inference (含可視化)
  
  ↓ (所有分支匯合)
  
entity_linker → knowledge_reasoner → image_explainer → report_generator
  ↓
END
```

---

### 🎨 G. 使用者介面 (`app/ui/`)

#### G1. 結構性 MRI 組件
**檔案：** `structural_mri_components.py`

**功能：**
- 專業臨床儀表板
- 中英雙語切換
- 結構化報告展示
- 腦區分析表格
- 臨床備註輸入

**UI 組件：**
1. **分析模式選擇器**
2. **臨床指標卡片**（診斷、預測、信心度）
3. **結構化報告展示**
   - 主要發現
   - 關鍵發現（結構變化、體積分析）
   - 臨床解釋（AD 指標、保護因子）
   - 建議（立即行動、監測、額外檢查）
4. **重要腦區表格**（排名、名稱、重要性、功能分類）
5. **臨床備註區**

---

## 核心處理管線

### 📊 處理管線 1: 結構性 MRI (CNN-RF)

```
1. 載入原始 MRI 影像
   ↓
2. 特徵提取 (AAL3 Atlas)
   - 灰質 (GM) 特徵
   - FA (各向異性分數)
   - MD (平均擴散率)
   ↓
3. 特徵標準化
   ↓
4. Random Forest 預測
   ↓
5. SHAP 局部解釋
   ↓
6. 腦區可視化
   ↓
7. 知識圖譜增強
   ↓
8. 結構化報告生成
```

### 🧠 處理管線 2: 功能性 fMRI

```
1. 載入 4D fMRI 數據
   ↓
2. 預處理 (切片/窗口)
   ↓
3. 深度學習推理
   ↓
4. 層選擇與驗證
   ↓
5. GradCAM 激活圖
   ↓
6. 腦區映射 (AAL3)
   ↓
7. 實體連結
   ↓
8. 知識圖譜增強
   ↓
9. 視覺解釋
   ↓
10. 敘述性報告生成
```

---

## 資料流與整合

### 輸入資料格式

**結構性 MRI:**
- 格式：NIfTI (.nii.gz)
- 類型：T1-weighted
- 空間：MNI152 或原始空間（自動重採樣）
- 解析度：1mm 或 2mm

**功能性 fMRI:**
- 格式：NIfTI (.nii.gz)
- 類型：BOLD 4D
- 預處理：建議已完成運動校正、時間校正

### 輸出資料

**可視化：**
- 特徵重要性圖 (PNG)
- 3D 腦區可視化 (PNG)
- GradCAM 熱力圖 (NIfTI + PNG)

**報告：**
- 結構化 JSON 報告
- 中英雙語文字報告
- 臨床建議

**資料：**
- ROI 特徵值 (CSV)
- 特徵重要性 (CSV)
- SHAP 值 (CSV)

---

## 技術棧

### 深度學習框架
- **PyTorch**: 深度學習模型
- **torchvision**: 影像處理

### 神經影像處理
- **nibabel**: NIfTI 檔案讀寫
- **nilearn**: 神經影像分析與可視化
- **ANTsPy**: 影像配準（可選）

### 機器學習
- **scikit-learn**: Random Forest、StandardScaler
- **SHAP**: 模型可解釋性

### 知識圖譜
- **Neo4j**: 圖資料庫
- **py2neo**: Python Neo4j 驅動

### LLM 整合
- **LangChain**: LLM 編排
- **LangGraph**: 工作流管理
- **boto3**: AWS Bedrock
- **ollama**: 本地 LLM

### UI 框架
- **Streamlit**: Web 介面
- **matplotlib** + **seaborn**: 資料可視化
- **pandas**: 資料處理

---

## 模型資訊

### 已訓練模型

**1. Random Forest (傳統 ML)**
- 路徑：`model/ml/final/`
- 特徵：32 AAL ROIs
- 準確率：75.4% (5-fold CV)
- 訓練資料：ADNI (65 subjects)

**2. CNN-RF (混合模型)**
- 路徑：`model/cnn_rf/`
- 特徵：多模態 (GM + FA + MD)
- 模型：
  - `rf_model_NC_vs_AD.joblib` (二分類)
  - `rf_model_NC_vs_AD_GM_only.joblib` (僅灰質)

**3. 3D CNN (fMRI)**
- 路徑：`model/cnn_3d/`
- 架構：3D 卷積網絡
- 輸入：4D fMRI (時序數據)

**4. ShuffleNet (2D CNN)**
- 路徑：`model/shufflenet/`
- 架構：ShuffleNet V2
- 輸入：2D 切片 (10 slices)

**5. CapsNet (膠囊網絡)**
- 路徑：`model/capsnet/` (腳本中)
- 架構：3D 膠囊網絡 + RNN
- 輸入：3D fMRI 窗口

---

## 資料集

### 訓練資料
- **ADNI** (Alzheimer's Disease Neuroimaging Initiative)
- **Cardinal Tien Hospital** (天主教耕莘醫院)

### 資料結構
```
data/
├── MRI_processed/     # 結構性 MRI (已預處理)
│   ├── AD/           # 阿茲海默症患者
│   ├── MCI/          # 輕度認知障礙
│   └── NC/           # 正常對照組
├── fMRI/             # 功能性 fMRI
│   ├── AD/
│   └── CN/
├── aal3/             # AAL3 腦區圖譜
└── templates/        # MNI152 模板
```

---

## 知識圖譜

### 圖譜結構

**節點類型：**
- `Region`: 腦區 (166 個 AAL3 區域)
- `YeoNetwork`: Yeo 7-network
- `Function`: 腦區功能

**關係類型：**
- `BELONGS_TO`: 腦區 → 神經網絡
- `HAS_FUNCTION`: 腦區 → 功能
- `CONNECTED_TO`: 腦區 ↔ 腦區 (結構連接)

**屬性：**
- `Region.ad_associated`: 是否與 AD 相關
- `Region.hemisphere`: 半球 (Left/Right)
- `Function.category`: 功能類別

### 資料來源
- AAL3 Atlas
- Yeo 7-Network Parcellation
- 神經科學文獻

---

## 可解釋性技術

### 1. SHAP (SHapley Additive exPlanations)
- **用途**：解釋 Random Forest 預測
- **輸出**：每個特徵對預測的貢獻值
- **優勢**：局部解釋，針對單一樣本

### 2. 特徵重要性
- **用途**：全局模型解釋
- **輸出**：特徵重要性排名
- **優勢**：簡單直觀

### 3. GradCAM
- **用途**：深度學習模型視覺化
- **輸出**：激活熱力圖
- **優勢**：空間定位

### 4. 知識圖譜增強
- **用途**：提供神經科學背景知識
- **輸出**：腦區功能、網絡歸屬
- **優勢**：臨床可解釋性

---

## 系統優勢

### 🎯 臨床價值
1. **多模態分析**：整合結構與功能影像
2. **可解釋性**：不只給預測，還解釋原因
3. **知識增強**：結合神經科學知識
4. **自動化報告**：節省醫師時間

### 🔬 技術優勢
1. **模組化設計**：易於擴展與維護
2. **多模型支援**：可根據需求選擇模型
3. **工作流管理**：LangGraph 確保流程可靠
4. **隱私保護**：支援本地 LLM

### 📊 研究價值
1. **端到端管線**：從影像到報告全自動
2. **可重現性**：標準化處理流程
3. **可擴展性**：易於添加新模型/新功能

---

## 未來發展方向

### 短期目標
- [ ] 支援 MCI (輕度認知障礙) 三分類
- [ ] 整合更多腦區圖譜 (Brainnetome, Schaefer)
- [ ] 優化 SHAP 計算效能
- [ ] 添加縱向追蹤分析

### 中期目標
- [ ] 多模態融合模型 (sMRI + fMRI + PET)
- [ ] 預後預測 (MCI → AD 轉化風險)
- [ ] 亞型分類 (典型 AD vs 非典型 AD)
- [ ] 臨床試驗整合

### 長期目標
- [ ] 聯邦學習 (多中心協作)
- [ ] 實時推理 (邊緣計算)
- [ ] 個人化治療建議
- [ ] FDA/TFDA 認證

---

## 系統限制

### 技術限制
1. **資料需求**：需要高品質的 MRI 影像
2. **計算資源**：深度學習模型需要 GPU
3. **模型泛化**：訓練資料主要來自 ADNI（西方人群）

### 臨床限制
1. **輔助診斷**：不能取代醫師判斷
2. **單一時間點**：缺乏縱向追蹤
3. **共病影響**：其他神經疾病可能影響結果

### 法規限制
1. **未經認證**：尚未通過醫療器材認證
2. **研究用途**：目前僅供研究使用
3. **資料隱私**：需符合當地法規

---

## 使用建議

### 適用場景
✅ 研究用途
✅ 臨床輔助診斷
✅ 教學示範
✅ 演算法開發

### 不適用場景
❌ 單獨作為診斷依據
❌ 緊急醫療決策
❌ 未經驗證的人群

### 最佳實踐
1. **資料品質**：確保 MRI 影像品質良好
2. **臨床整合**：結合臨床評估與生物標記
3. **持續驗證**：定期驗證模型表現
4. **透明溝通**：向患者說明 AI 輔助的角色

---

## 參考文獻

### 方法學
- AAL3 Atlas: Rolls et al., 2020
- Yeo 7-Network: Yeo et al., 2011
- SHAP: Lundberg & Lee, 2017
- GradCAM: Selvaraju et al., 2017

### 資料集
- ADNI: http://adni.loni.usc.edu/
- Cardinal Tien Hospital (合作資料)

### 相關論文
- (待補充：系統相關發表)

---

## 聯絡資訊

**開發團隊：** [待補充]
**技術支援：** [待補充]
**問題回報：** [待補充]

---

**最後更新：** 2024-11-19
**版本：** 1.0.0
**文件語言：** 繁體中文 + English
