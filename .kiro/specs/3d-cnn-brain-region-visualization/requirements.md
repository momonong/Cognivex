# 需求文件

## 簡介

本功能旨在為 3D CNN 模型提供完整的可解釋性分析工具，幫助研究人員理解模型在判斷阿茲海默症 (AD) 時關注的腦區。透過視覺化模型的 activation 和 attention maps，並將其映射到標準腦區圖譜，研究人員可以驗證模型的決策是否符合臨床知識，並發現潛在的生物標記。

## 術語表

- **3D CNN Model**: 用於分類 AD 和 NC (正常對照) 的三維卷積神經網路模型
- **Activation Map**: 神經網路中間層的激活值，反映模型對輸入特徵的響應
- **Grad-CAM**: 梯度加權類別激活映射，一種視覺化技術，顯示模型關注的區域
- **Brain Atlas**: 標準腦區圖譜，如 AAL (Automated Anatomical Labeling) 或 Harvard-Oxford Atlas
- **NIfTI**: 神經影像常用的檔案格式 (.nii 或 .nii.gz)
- **Heatmap**: 熱圖，用顏色強度表示數值大小的視覺化方式
- **Brain Region Mapping**: 將 activation 值映射到具體腦區名稱的過程
- **Ensemble Model**: 集成模型，由多個獨立訓練的模型組成

## 需求

### 需求 1: 多層 Activation 擷取與儲存

**使用者故事:** 作為研究人員，我想要擷取模型不同層的 activation maps，以便分析模型在不同抽象層次上關注的特徵

#### 驗收標準

1. WHEN 使用者指定目標層名稱，THE 3D CNN Model SHALL 在前向傳播時擷取該層的 activation tensor
2. THE 3D CNN Model SHALL 支援同時擷取多個層的 activation maps
3. THE 3D CNN Model SHALL 將擷取的 activation 儲存為 .pt 格式，包含 tensor 資料和 metadata (層名稱、shape、受試者 ID)
4. WHERE 使用者需要 Grad-CAM 分析，THE 3D CNN Model SHALL 同時儲存 activation 和對應的 gradient tensors
5. THE 3D CNN Model SHALL 在擷取過程中保持模型推論結果的正確性

### 需求 2: Grad-CAM 熱圖生成與優化

**使用者故事:** 作為研究人員，我想要生成高品質的 Grad-CAM 熱圖，以便清楚看到模型關注的腦區

#### 驗收標準

1. WHEN 提供 activation 和 gradient tensors，THE Visualization System SHALL 計算 Grad-CAM 熱圖
2. THE Visualization System SHALL 將 Grad-CAM 熱圖上採樣至原始 NIfTI 影像的空間解析度
3. THE Visualization System SHALL 保持原始 NIfTI 的 affine 矩陣，確保空間對齊正確
4. WHERE 使用者使用集成模型，THE Visualization System SHALL 平均多個模型的熱圖並計算信心區間
5. THE Visualization System SHALL 提供可調整的閾值參數，過濾低激活值區域

### 需求 3: 腦區映射與量化分析

**使用者故事:** 作為研究人員，我想要知道模型具體關注哪些腦區，以及每個腦區的重要性分數

#### 驗收標準

1. THE Brain Region Mapper SHALL 載入標準腦區圖譜 (AAL 或 Harvard-Oxford)
2. WHEN 提供 Grad-CAM 熱圖 NIfTI，THE Brain Region Mapper SHALL 將熱圖與腦區圖譜進行空間配準
3. THE Brain Region Mapper SHALL 計算每個腦區的平均 activation 強度
4. THE Brain Region Mapper SHALL 輸出排序後的腦區重要性列表，包含腦區名稱和分數
5. THE Brain Region Mapper SHALL 將結果儲存為 CSV 和 JSON 格式

### 需求 4: 互動式視覺化介面

**使用者故事:** 作為研究人員，我想要在瀏覽器中互動式地查看熱圖疊加在原始影像上，以便更直觀地理解結果

#### 驗收標準

1. THE Visualization Interface SHALL 在網頁瀏覽器中顯示 3D 腦部影像
2. THE Visualization Interface SHALL 支援疊加顯示 Grad-CAM 熱圖，並可調整透明度
3. THE Visualization Interface SHALL 提供三個正交切面視圖 (矢狀面、冠狀面、軸向面)
4. WHEN 使用者點擊影像上的某個點，THE Visualization Interface SHALL 顯示該點所屬的腦區名稱和 activation 值
5. THE Visualization Interface SHALL 支援匯出當前視圖為圖片檔案

### 需求 5: 批次處理與報告生成

**使用者故事:** 作為研究人員，我想要批次處理多個受試者的資料，並生成統計報告

#### 驗收標準

1. THE Batch Processing System SHALL 接受包含多個 NIfTI 檔案的資料夾路徑
2. THE Batch Processing System SHALL 對每個受試者執行完整的分析流程 (Grad-CAM 生成、腦區映射)
3. THE Batch Processing System SHALL 顯示處理進度條和預估剩餘時間
4. THE Batch Processing System SHALL 生成群組層級的統計報告，包含最常被關注的腦區
5. WHERE 處理過程中發生錯誤，THE Batch Processing System SHALL 記錄錯誤訊息並繼續處理其他檔案

### 需求 6: 配置管理與可重現性

**使用者故事:** 作為研究人員，我想要記錄所有分析參數，以便實驗結果可重現

#### 驗收標準

1. THE Configuration System SHALL 從 .env 或 YAML 檔案讀取所有分析參數
2. THE Configuration System SHALL 驗證參數的有效性 (如檔案路徑存在、數值範圍合理)
3. THE Configuration System SHALL 在輸出資料夾中儲存使用的完整配置
4. THE Configuration System SHALL 記錄模型版本、程式碼版本和執行時間戳
5. THE Configuration System SHALL 支援從先前的配置檔案重新執行分析
