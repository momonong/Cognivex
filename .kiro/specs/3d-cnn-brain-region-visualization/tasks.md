# 實作計畫

- [x] 1. 建立專案結構和配置系統









  - 建立 `app/core/xai/` 目錄結構
  - 實作 `ConfigManager` 類別，支援 YAML 配置載入和驗證
  - 建立預設配置檔案 `config/xai_config.yaml`
  - 實作配置儲存到輸出目錄的功能
  - _需求: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 2. 實作 Activation Extractor





  - [x] 2.1 建立 `ActivationExtractor` 類別


    - 實作 PyTorch hooks 註冊機制
    - 支援多層同時擷取 activation 和 gradient
    - 實作 `extract()` 方法執行前向和反向傳播
    - _需求: 1.1, 1.2, 1.4_
  


  - [x] 2.2 實作資料儲存功能





    - 實作 `save_to_disk()` 方法，儲存為 .pt 格式
    - 包含 metadata (subject_id, layer_name, shape, timestamp)
    - 驗證儲存和載入的正確性
    - _需求: 1.3, 1.5_

- [x] 3. 重構 Grad-CAM Generator





  - [x] 3.1 建立 `GradCAMGenerator` 類別


    - 從現有 `app/core/cnn_3d/xai.py` 提取核心邏輯
    - 實作 `generate_single_model()` 方法
    - 實作 `generate_ensemble()` 方法，支援 5-fold 集成
    - _需求: 2.1, 2.4_
  
  - [x] 3.2 實作熱圖處理和儲存


    - 實作上採樣到原始解析度的功能
    - 實作可調整閾值的過濾機制
    - 實作 `save_as_nifti()` 方法，保持正確的 affine 矩陣
    - _需求: 2.2, 2.3, 2.5_

- [ ] 4. 實作 Brain Region Mapper
  - [ ] 4.1 建立 `BrainRegionMapper` 類別和圖譜載入
    - 實作 `load_atlas()` 方法，載入 AAL3 NIfTI 和 XML 標籤
    - 解析 XML 檔案取得腦區名稱對應
    - 快取圖譜資料以提升效能
    - _需求: 3.1_
  
  - [ ] 4.2 實作影像配準功能
    - 使用 nilearn 實作 `register_to_atlas()` 方法
    - 處理不同空間解析度的熱圖
    - 驗證配準結果的空間對齊
    - _需求: 3.2_
  
  - [ ] 4.3 實作腦區分數計算
    - 實作 `compute_region_scores()` 方法
    - 支援多種聚合方法 (mean, max, weighted_mean)
    - 計算每個腦區的體素數量和百分比
    - _需求: 3.3_
  
  - [ ] 4.4 實作結果匯出
    - 實作 `export_results()` 方法
    - 匯出 CSV 格式 (region_id, region_name, scores)
    - 匯出 JSON 格式 (包含 metadata 和統計資訊)
    - _需求: 3.4, 3.5_

- [ ] 5. 實作 Quantitative Analyzer
  - [ ] 5.1 建立 `QuantitativeAnalyzer` 類別
    - 實作 `rank_regions()` 方法，返回 top-K 腦區
    - 實作排序和過濾邏輯
    - _需求: 3.4_
  
  - [ ] 5.2 實作統計分析功能
    - 實作 `compute_statistics()` 方法，計算群組統計
    - 計算 mean, std, confidence intervals
    - 實作 `compare_groups()` 方法，比較 AD vs NC
    - _需求: 5.4_
  
  - [ ]* 5.3 實作摘要報告生成
    - 實作 `generate_summary_report()` 方法
    - 生成文字格式的統計摘要
    - 包含 top regions 和關鍵發現
    - _需求: 5.4_

- [ ] 6. 實作 Visualization Engine
  - [ ] 6.1 建立 `VisualizationEngine` 類別
    - 實作 `plot_brain_slices()` 方法，使用 nilearn
    - 支援自訂切面座標和顯示模式
    - _需求: 4.3_
  
  - [ ] 6.2 實作多種視覺化方法
    - 實作 `plot_glass_brain()` 方法
    - 實作 `plot_region_bar_chart()` 方法，顯示 top regions
    - 支援自訂 colormap 和透明度
    - _需求: 4.2, 4.5_
  
  - [ ] 6.3 實作互動式視圖
    - 實作 `create_interactive_viewer()` 方法
    - 使用 plotly 或 nilearn 生成 HTML
    - 支援點擊顯示腦區資訊
    - _需求: 4.1, 4.4_

- [ ] 7. 實作 Batch Processor
  - [ ] 7.1 建立 `BatchProcessor` 類別
    - 實作 `process_directory()` 方法
    - 遞迴掃描資料夾中的 NIfTI 檔案
    - 整合所有處理步驟 (Grad-CAM → 腦區映射 → 視覺化)
    - _需求: 5.1, 5.2_
  
  - [ ] 7.2 實作進度追蹤和錯誤處理
    - 使用 tqdm 顯示進度條和預估時間
    - 實作錯誤記錄機制，繼續處理其他檔案
    - 返回處理結果摘要 (成功/失敗數量)
    - _需求: 5.3, 5.5_
  
  - [ ] 7.3 實作群組報告生成
    - 實作 `generate_group_report()` 方法
    - 聚合所有受試者的腦區分數
    - 生成群組統計和視覺化
    - _需求: 5.4_

- [ ] 8. 建立 Streamlit 互動式介面
  - [ ] 8.1 建立 UI 基本結構
    - 建立 `app/ui/xai_viewer.py` 檔案
    - 實作側邊欄配置區 (檔案上傳、參數設定)
    - 實作主顯示區的版面配置
    - _需求: 4.1, 4.2, 4.3_
  
  - [ ] 8.2 整合分析流程
    - 實作 "Run Analysis" 按鈕的回調函式
    - 整合所有處理元件 (Grad-CAM → Mapper → Viz)
    - 實作即時進度顯示
    - _需求: 5.3_
  
  - [ ] 8.3 實作結果顯示
    - 顯示 Grad-CAM 熱圖 (切片視圖)
    - 顯示 top brain regions 長條圖
    - 嵌入互動式 3D 視圖 (HTML)
    - 顯示腦區詳細資訊 DataFrame
    - _需求: 4.1, 4.2, 4.3, 4.4_
  
  - [ ] 8.4 實作結果匯出功能
    - 實作下載按鈕 (CSV, JSON, PNG, NIfTI)
    - 支援匯出當前視圖為圖片
    - _需求: 4.5_

- [ ] 9. 實作日誌和錯誤處理系統
  - 配置 Python logging 模組
  - 實作統一的錯誤處理機制
  - 記錄所有關鍵操作和錯誤到日誌檔案
  - 提供清楚的使用者錯誤訊息
  - _需求: 6.4_

- [ ] 10. 建立命令列介面
  - 建立 `scripts/run_xai_analysis.py` 腳本
  - 支援單一檔案和批次處理模式
  - 使用 argparse 處理命令列參數
  - 整合配置檔案載入
  - _需求: 5.1, 5.2, 6.1_

- [ ]* 11. 撰寫測試
  - [ ]* 11.1 撰寫單元測試
    - 測試 `ActivationExtractor` 的 hook 機制
    - 測試 `GradCAMGenerator` 的熱圖計算
    - 測試 `BrainRegionMapper` 的分數計算
    - 測試 `ConfigManager` 的驗證邏輯
  
  - [ ]* 11.2 撰寫整合測試
    - 測試端到端流程 (NIfTI → 腦區分數)
    - 測試批次處理功能
    - 測試視覺化生成
  
  - [ ]* 11.3 建立測試 fixtures
    - 建立 mock 模型權重
    - 建立 mock NIfTI 檔案
    - 建立測試配置檔案

- [ ] 12. 撰寫文件和範例
  - [ ] 12.1 更新 README
    - 新增 XAI 分析功能說明
    - 提供使用範例和命令
    - 說明輸出格式
  
  - [ ] 12.2 建立使用者指南
    - 建立 `docs/xai_guide.md`
    - 說明配置選項
    - 提供常見問題解答
  
  - [ ]* 12.3 建立 API 文件
    - 為所有公開類別和方法撰寫 docstrings
    - 使用 Sphinx 生成 API 文件
  
  - [ ] 12.4 建立範例 notebook
    - 建立 `examples/xai_analysis_example.ipynb`
    - 展示完整的分析流程
    - 包含視覺化範例

- [ ] 13. 整合到現有系統
  - 更新 `requirements.txt` 新增必要套件
  - 整合到現有 `app.py` Streamlit 介面
  - 確保與現有 LangGraph workflow 相容
  - 更新專案目錄結構文件
  - _需求: 6.1_
