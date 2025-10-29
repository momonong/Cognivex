# 應用程式架構與工作流程文件

本文件詳細說明了 `app` 目錄的程式碼架構、工作流程、核心管線以及各模組的詳細功能，旨在幫助開發者理解和修改此 fMRI 分析應用程式。

## 1. 高層架構 (High-Level Architecture)

應用程式遵循一個基於代理 (Agent) 和圖 (Graph) 的架構，將複雜的 fMRI 分析任務分解為一系列獨立但相互關聯的節點。每個節點執行一個特定的任務，並透過一個共享的狀態 (State) 來傳遞資料。

主要目錄結構如下：

-   `app/graph`: 定義了整個工作流程的狀態 (`state.py`) 和執行圖 (`workflow.py`)。這是應用的「大腦」，負責調度所有其他模組。
-   `app/agents`: 包含一系列「代理」或「節點」，每個節點都是圖中的一個執行單元，負責執行具體的業務邏輯，例如模型推論、影像後處理、知識圖譜查詢等。
-   `app/core`: 包含了應用程式的核心功能和工具，這些是被 `agents` 所呼叫的底層實作。它分為：
    -   `fmri_processing`: fMRI 影像處理的核心管線，包括從模型推論到特徵提取、視覺化的所有步驟。
    -   `knowledge_graph`: 與知識圖譜互動的工具，包括實體連結和查詢。
    -   `vision`: 視覺分析工具，用於解釋影像。
-   `app/services`: 封裝了與外部服務的連接，例如不同的 LLM 供應商 (Gemini, Bedrock) 和 Neo4j 資料庫。

## 2. 工作流程 (Workflow)

工作流程由 `langgraph` 函式庫驅動，定義在 `app/graph/workflow.py` 中。

### 2.1. 狀態 (AgentState)

`app/graph/state.py` 中的 `AgentState` 是一個 `TypedDict`，定義了在整個工作流程中傳遞的所有資料。它包含了：

-   **Inputs**: 啟動流程所需的初始資料，如 `subject_id`, `fmri_scan_path`。
-   **Intermediate Data**: 節點之間傳遞的中繼資料，如 `validated_layers`, `clean_region_names`。
-   **Final Outputs**: 最終產生的結果，如 `classification_result`, `activated_regions`, `generated_reports`。
-   **System & Tracing**: 用於日誌和錯誤追蹤的 `trace_log` 和 `error_log`。

### 2.2. 執行圖 (Execution Graph)

`app/graph/workflow.py` 中定義了節點的執行順序：

```
START -> inference -> filtering -> post_processing -> entity_linker -> knowledge_reasoner -> image_explainer -> report_generator -> END
```

1.  **inference**: 執行深度學習模型推論，進行初步分類。
2.  **filtering**: (此節點在目前版本中似乎未被使用，但可能用於過濾不重要的模型層)。
3.  **post_processing**: 對模型推論的結果（特別是激活圖）進行後處理，包括重採樣、腦區激活分析和視覺化。
4.  **entity_linker**: 將後處理得到的「髒」腦區名稱與知識圖譜中的標準名稱進行對齊（實體連結）。
5.  **knowledge_reasoner**: 使用對齊後的「乾淨」腦區名稱，從知識圖譜中查詢相關的醫學知識（如相關網絡、功能）。
6.  **image_explainer**: 呼叫多模態 LLM，結合視覺化的激活圖和結構化數據，生成對影像的解釋。
7.  **report_generator**: 綜合所有資訊（分類結果、腦區分析、影像解釋），生成最終的臨床報告（中英文）。

## 3. 核心管線：fMRI 處理 (Core Pipeline: fMRI Processing)

最核心的處理邏輯位於 `app/core/fmri_processing/generic_pipeline_steps.py` 中的 `GenericInferencePipeline` 類別。這個管線被 `inference` 代理節點所呼叫，並執行以下一系列複雜步驟來實現 XAI (Explainable AI) 功能。

### `GenericInferencePipeline` 執行流程

這個類別整合了從模型加載到最終視覺化產出的完整流程。

1.  **初始化 (`__init__`)**:
    -   根據傳入的 `model_config` (如 "papermodel")，載入對應的 `ModelConfig`，其中包含了模型類型、輸入形狀、後處理所需的各種路徑 (MNI 模板、腦圖譜等)。
    -   使用 `ModelFactory` 創建對應的 `ModelAdapter` (例如 `PaperModelAdapter`)，這個適配器封裝了模型特定的預處理和後處理邏輯。

2.  **`run_full_pipeline` (主方法)**:
    -   **`inspect_and_select_layers`**:
        -   呼叫 `inspector.inspect_torch_model` 來遍歷模型的所有層。
        -   呼叫 `choose_layer.select_visualization_layers`，它會將層列表和一個策略 (例如 `"shufflenet_focused_v3"`) 發送給 LLM，讓 LLM 根據預設的指令選擇最適合用於視覺化的層。
    -   **`prepare_model`**:
        -   加載預訓練的模型權重 (`.pth` 檔案)。
        -   將模型設置為評估模式 (`.eval()`)。
    -   **`run_inference_with_hooks`**:
        -   **數據預處理**: 使用 `ModelAdapter` 的 `preprocess_data` 方法將輸入的 NIfTI 檔案轉換為模型所需的 Tensor 格式。
        -   **掛載鉤子 (Hooks)**:
            -   呼叫 `attach_hook.prepare_model_with_hooks` 在選定的目標層上註冊「前向鉤子」(forward hook)，用於在模型前向傳播時捕獲該層的「激活」(activations)。
            -   呼叫 `attach_hook.attach_gradient_hooks` 註冊「後向鉤子」(backward hook)，用於在反向傳播時捕獲「梯度」(gradients)。
        -   **執行推論**:
            -   執行模型的前向傳播 (`model(inputs)`)，觸發前向鉤子，捕獲激活。
            -   為了計算 Grad-CAM，需要進行反向傳播。程式會選定目標類別的分數，並呼叫 `.backward()`，觸發後向鉤子，捕獲梯度。
        -   **儲存結果**: 將捕獲的激活和梯度保存為 `.pt` 檔案。
        -   **清理鉤子**: 移除所有鉤子以防內存洩漏。
    -   **`run_post_processing`**:
        -   **`act_to_nii.activation_and_gradient_to_nifti`**:
            -   加載前一步保存的激活和梯度。
            -   計算 **Grad-CAM** 熱圖。
            -   將 2D 的 Grad-CAM 熱圖切片重新投影回原始 3D NIfTI 影像的空間，生成一個「原生空間 (native space)」的 3D 熱圖 (`.nii.gz`)。
        -   **`spatial_normalizer.normalize_native_heatmap_to_mni_accurate_masked`**:
            -   使用 **ANTsPy** 工具庫，將原生空間的熱圖進行「空間標準化」，對齊到標準的 **MNI 空間**。這是一個非常耗時但關鍵的步驟，確保結果可以和標準腦圖譜比較。
        -   **`resample.resample_activation_to_atlas`**:
            -   將 MNI 空間的熱圖進一步「重採樣」，使其網格與目標腦圖譜 (如 AAL3) 完全對齊。
        -   **`brain_map.analyze_brain_activation`**:
            -   使用最終對齊的熱圖和腦圖譜，分析每個腦區的平均激活強度，生成一個包含「腦區名稱」和「激活分數」的表格 (DataFrame)。
        -   **`visualize.visualize_gradcam_2d`**:
            -   生成 2D 的 Grad-CAM 視覺化圖片 (PNG)，將熱圖疊加在原始的 MRI 切片上。

## 4. 代理 (Agents) 詳解

-   `agents/inference.py`: 呼叫 `GenericInferencePipeline` 來執行模型推論和初步分類。
-   `agents/postprocessing.py`: 執行後處理步驟，分析激活圖，找出顯著激活的腦區，並生成視覺化圖片。它還會解析腦區名稱中的半球資訊 (左/右)。
-   `agents/entity_linking.py`:
    -   從 `postprocessing` 獲取可能不規範的腦區名稱列表。
    -   呼叫 `core/knowledge_graph/entity_linker.py` 中的 `entity_linker_tool`。
    -   `entity_linker_tool` 會從 Neo4j 數據庫獲取所有標準的腦區名稱，然後使用 LLM 將「髒」列表與標準列表進行模糊匹配，返回一個「乾淨」的、保證存在於數據庫中的名稱列表。
-   `agents/knowledge_reasoning.py`:
    -   接收「乾淨」的腦區名稱列表。
    -   呼叫 `core/knowledge_graph/query_engine.py` 中的 `graphrag` 工具。
    -   `graphrag` 會執行一個預先定義好的 Cypher 查詢，從 Neo4j 中批量獲取這些腦區關聯的「神經網絡」和「已知功能」。
    -   將查詢到的知識合併回 `AgentState` 的 `activated_regions` 中。
-   `agents/image_explainer.py`:
    -   呼叫 `core/vision/explain_tool.py` 中的 `explain_activation_map`。
    -   此工具會將視覺化圖片、結構化的腦區分析數據、以及整體分類結果一起發送給多模態 LLM (如 Gemini Pro Vision)。
    -   LLM 被指示以結構化數據為「事實基礎」，結合圖片進行視覺描述，生成一段關於激活模式的臨床解釋。
-   `agents/report_generator.py`:
    -   收集 `AgentState` 中的所有最終資訊。
    -   構建一個詳細的 Prompt，要求 LLM 扮演神經放射科醫生的角色，綜合所有資訊撰寫一份包含「主要發現」、「腦活動模式解讀」、「與神經學知識的關聯」和「結論」四個部分的最終報告。
    -   分別生成英文報告和中文翻譯。

## 5. 服務 (Services) 詳解

-   `services/llm_providers/`: 這是一個 LLM 的路由/適配器層。
    -   `__init__.py`: 提供了 `llm_response` 和 `llm_image_response` 兩個統一的接口。它們會根據傳入的 `llm_provider` 參數（如 "gemini", "aws_bedrock"）將請求分派給對應的模組。
    -   `gemini.py`, `bedrock.py`, `ollama.py`: 每個文件都實現了與特定 LLM 服務 API 的對接細節，包括如何構建請求、處理認證和解析響應。這種設計使得切換或增加新的 LLM 供應商變得非常容易。
-   `services/neo4j_connector.py`:
    -   提供一個 `get_neo4j_driver` 函式，用於從環境變數中讀取連接配置（URI, 用戶名, 密碼），並創建一個全局可用的 Neo4j 數據庫驅動實例。所有與數據庫的交互都通過這個驅動進行。

## 6. 如何修改與擴展

-   **替換模型**:
    1.  在 `app/core/fmri_processing/model_config.py` 中，為你的新模型創建一個新的 `ModelAdapter` (繼承自 `BaseModelAdapter`)。
    2.  實現 `create_model`, `preprocess_data`, `get_layer_selection_strategy`, 和 `postprocess_prediction` 四個抽象方法。
    3.  在 `ModelFactory` 中註冊你的新適配器。
    4.  在 `get_config_by_name` 中為你的模型添加一個新的配置項。
    5.  在啟動工作流程時，於 `initial_state` 中傳入你的新模型名稱 (`model_name`)。
-   **修改 Prompt**:
    -   每個代理或核心工具中與 LLM 交互的部分，其 Prompt (或 System Instruction) 都被明確定義為字串變數（如 `report_generator.py` 中的 `synthesis_prompt`，或 `entity_linker.py` 中的 `prompt`）。直接修改這些字串即可調整 LLM 的行為。
-   **增加新的處理節點**:
    1.  在 `app/agents/` 目錄下創建一個新的 Python 檔案，定義你的新節點函式，其簽名必須是 `(state: AgentState) -> dict`。
    2.  在 `app/graph/workflow.py` 中，導入你的新節點函式。
    3.  使用 `workflow.add_node("your_node_name", your_node_function)` 將其添加到圖中。
    4.  修改 `workflow.add_edge(...)` 來將你的新節點連接到現有流程中。
-   **修改知識圖譜查詢**:
    -   直接修改 `app/core/knowledge_graph/query_engine.py` 中的 `CYPHER_TEMPLATE` 字串即可更改從 Neo4j 中提取的數據。
