# 機器學習腦區分析

使用 Random Forest 分析阿茲海默症相關的重要腦區

## 📁 檔案結構

```
scripts/ml/          # 訓練和測試腳本
├── train_ml_model.py    # 訓練模型
├── predict.py           # 預測腳本
└── README.md            # 說明文件

model/ml/            # 訓練好的模型
├── rf_model.pkl         # Random Forest 模型
└── scaler.pkl           # 特徵標準化器

output/ml/           # 訓練結果
├── roi_importance.csv   # ROI 重要性排名
├── training_results.csv # 訓練集預測結果
└── training_summary.csv # 訓練摘要
```

## 📋 腳本說明

### analyze_feature_importance.py
**分析模型是否真的學習到阿茲海默症相關的關鍵腦區特徵**

這個腳本使用兩種方法計算特徵重要性：
1. **Gini Importance** - Random Forest 內建方法
2. **Permutation Importance** - 更可靠的方法（隨機打亂特徵值後觀察效能下降）

**關鍵腦區定義：**
- 海馬迴相關（Hippocampus, Amygdala）
- 內嗅皮質（Entorhinal cortex）
- 顳葉（Temporal lobe）
- 頂葉（Parietal lobe）
- 扣帶迴（Cingulate cortex）

**輸出檔案：**
- `feature_importance_report.txt` - 詳細分析報告
- `feature_importance_details.csv` - 所有特徵的重要性數據
- `top_features_importance.png` - Top 30 特徵視覺化
- `region_category_importance.png` - 各腦區類別重要性
- `critical_vs_other.png` - 關鍵腦區 vs 其他腦區比較

### batch_predict.py
批次預測多個影像並生成詳細報告。

## 🚀 快速開始

### 1. 訓練模型

```bash
python scripts/ml/train_ml_model.py
```

**輸出**：
- `model/ml/rf_model.pkl` - 訓練好的模型
- `model/ml/scaler.pkl` - 特徵標準化器
- `output/ml/roi_importance.csv` - ROI 重要性排名
- `output/ml/training_results.csv` - 訓練結果

**預期結果**：
```
交叉驗證準確率: 74%
訓練集準確率: 95%+
最重要的腦區: Cingulum_Post_R (後扣帶迴右側)
```

### 2. 分析特徵重要性（確認模型學到正確的腦區）

```bash
python scripts/ml/analyze_feature_importance.py
```

**這個步驟很重要！** 它會告訴你：
- ✅ 模型是否真的依賴關鍵腦區（海馬迴、內嗅皮質等）
- ✅ 還是只是過擬合到一些無關的特徵
- ✅ Top 20 重要特徵中有多少來自關鍵腦區

**預期結果：**
```
關鍵腦區重要性佔比: 88.98%  ← 表示模型學到正確特徵
Top 20 中來自關鍵腦區: 17/20  ← 超過一半來自關鍵腦區
```

### 3. 比較 24 ROIs vs 全部 116 ROIs（可選）

如果你想驗證 24 個精選腦區是否足夠，可以進行完整比較：

**步驟 1：提取所有 116 個 AAL ROI 特徵**
```bash
python scripts/ml/extract_all_roi_features.py
```

這會從所有 MRI 影像中提取全部 116 個 AAL 腦區的特徵。

**步驟 2：比較不同特徵集**
```bash
python scripts/ml/compare_real_features.py
```

這會比較：
- 24 個精選 ROIs
- 全部 116 個 ROIs（Random Forest）
- 全部 116 個 ROIs（L1 正則化）
- Top 30 ROIs（單變量選擇）
- Top 30 ROIs（互信息選擇）

**輸出：**
- 詳細的效能比較報告
- 視覺化圖表
- 過擬合分析
- 建議使用哪種方法

### 4. 訓練最終模型（混合方案）✨

**推薦使用最終版本！**

```bash
python scripts/ml/train_final_model.py
```

這會訓練使用 **32 個 ROIs 的混合模型**：
- 24 個文獻選擇的 ROIs（領域知識）
- + 8 個數據驅動的 ROIs（從 Top 30 分析）
- = 最佳平衡：效能 + 可解釋性

**輸出：**
- `model/ml/final/final_model.pkl` - 最終模型
- `model/ml/final/final_scaler.pkl` - 特徵標準化器
- `output/ml/final_model/final_model_report.txt` - 詳細報告
- `output/ml/final_model/final_model_analysis.png` - 視覺化分析

**預期結果：**
```
CV Accuracy: 75.4% ± 5.8%
ROC-AUC: 80.1% ± 6.7%
Overfitting Gap: 0.246
```

### 5. 使用模型預測

```bash
python scripts/ml/predict.py --input path/to/image_T1.nii.gz
```

**範例**：
```bash
python scripts/ml/predict.py --input E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/AD/sub_0082_T1.nii.gz
```

**輸出**：
```
預測類別: AD (阿茲海默症)

信心度:
  NC (正常):      15.23%
  AD (阿茲海默症): 84.77%
```

## 📊 重要腦區

根據訓練結果，以下腦區對 AD 分類最重要：

1. **Cingulum_Post (後扣帶迴)** - 最重要
   - 與 Default Mode Network 相關
   - AD 早期就會受影響

2. **Temporal (顳葉)**
   - AD 的典型受損區域
   - 與記憶形成相關

3. **Hippocampus (海馬迴)**
   - AD 最經典的生物標記
   - 與短期記憶相關

## 🔬 技術細節

### 模型架構
- **演算法**: Random Forest
- **特徵**: 24 個重要 ROI 的平均強度
- **樣本**: NC (42) vs AD (23)
- **交叉驗證**: 5-fold Stratified K-Fold

### 特徵提取
1. 使用 AAL atlas 定義 ROI
2. 計算每個 ROI 的平均強度
3. 標準化特徵 (Z-score)

### 超參數
```python
n_estimators=500      # 樹的數量
max_depth=10          # 最大深度
min_samples_split=5   # 最小分割樣本數
class_weight='balanced'  # 處理類別不平衡
```

## 📈 效能指標

| 指標 | 數值 |
|------|------|
| 交叉驗證準確率 | 74% |
| 訓練集準確率 | 95%+ |
| NC 召回率 | 95%+ |
| AD 召回率 | 90%+ |

## 🎯 使用場景

### Demo 展示
```python
import joblib
import pandas as pd

# 載入模型
model = joblib.load("model/ml/rf_model.pkl")
scaler = joblib.load("model/ml/scaler.pkl")

# 載入訓練結果
results = pd.read_csv("output/ml/training_results.csv")

# 顯示預測正確的樣本
correct = results[results['correct'] == True]
print(f"預測正確: {len(correct)}/{len(results)}")
```

### 批次預測
```python
import glob

# 取得所有 T1 影像
images = glob.glob("data/**/*_T1.nii.gz", recursive=True)

# 批次預測
for img_path in images:
    prediction, probability = predict(img_path)
    print(f"{img_path}: {prediction} ({probability[1]:.2%})")
```

## 🔍 驗證結果

結果符合神經科學文獻：
- ✅ 後扣帶迴排名第一 (符合 AD 研究)
- ✅ 顳葉排名前列 (AD 典型受損區域)
- ✅ 海馬迴在前 10 (AD 經典生物標記)

## 📚 參考文獻

1. Posterior Cingulate Cortex in AD (Buckner et al., 2005)
2. Temporal Lobe Atrophy in AD (Jack et al., 2010)
3. Hippocampal Volume in AD (Dubois et al., 2014)

## 💡 下一步

1. **加入多模態特徵** (T1 + T2 + DWI)
2. **使用集成學習** (RF + GB + SVM)
3. **加入臨床資料** (年齡、性別、MMSE)
4. **整合到系統** (Web UI 或 API)

## 🐛 疑難排解

### 問題：找不到模型檔案
```
FileNotFoundError: 找不到模型檔案: model/ml/rf_model.pkl
```
**解決**：先執行 `python scripts/ml/train_ml_model.py` 訓練模型

### 問題：AAL atlas 下載失敗
**解決**：檢查網路連線，或手動下載 AAL atlas

### 問題：記憶體不足
**解決**：減少 `n_estimators` 或使用更少的 ROI

## 📞 聯絡

如有問題，請參考主專案的 README 或聯絡開發團隊。
