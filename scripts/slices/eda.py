import pandas as pd
from collections import Counter, defaultdict

# 讀取切片 metadata
slice_df = pd.read_csv("data/slices_metadata.csv")

# 主體檢查資訊
print("📊 總切片數量：", len(slice_df))
print("🧠 總 subject 數量（根據 subject_id）：", slice_df["subject_id"].nunique())

# 確保每個 fold 都有
folds = slice_df["fold"].unique()
print(f"\n🔍 偵測到 {len(folds)} 個 fold：{folds}")

# 每個 fold 的統計資訊
for fold in sorted(folds):
    fold_df = slice_df[slice_df["fold"] == fold]
    subjects = fold_df["subject_id"].unique()
    label_dist = Counter(fold_df["label"])
    
    print(f"\n=== Fold {fold} ===")
    print(f"📁 總切片數：{len(fold_df)}")
    print(f"👥 Subject 數：{len(subjects)}")
    print(f"📌 類別分布：{dict(label_dist)}")
    
    if len(label_dist) < 2:
        print("⚠️ 僅包含單一類別，無法做有效訓練與驗證！")
    elif min(label_dist.values()) < 20:
        print("⚠️ 類別不平衡，驗證結果可能不穩！")

# 檢查是否有 subject 同時出現在多個 fold（防洩漏）
subject_folds = defaultdict(set)
for row in slice_df.itertuples():
    subject_folds[row.subject_id].add(row.fold)

leak_risk = {s: f for s, f in subject_folds.items() if len(f) > 1}
if leak_risk:
    print("\n🚨 有資料洩漏風險的 subject（同時出現在多個 fold）：")
    for sid, folds in leak_risk.items():
        print(f" - {sid}: fold {sorted(list(folds))}")
else:
    print("\n✅ 沒有發現 subject 出現在多個 fold，資料切分乾淨。")
