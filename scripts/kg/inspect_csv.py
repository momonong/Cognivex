import pandas as pd

df = pd.read_csv("output/kg/knowledge_graph_brain_regions.csv")

# 檢查是否有空或無效值
print("aRegion null:", df['aRegion'].isnull().sum())
print("aRegion empty:", (df['aRegion'].str.strip() == '').sum())

# 檢查 Region 對 Network 數量
print("🔗 Region ➝ Network pair 數量:", df[['aRegion', 'Yeo_Network']].drop_duplicates().shape[0])

# 檢查總筆數與去重後數量
print("📊 原始筆數:", len(df))
print("🧹 去重後筆數:", df.drop_duplicates().shape[0])

# 看看 AD_Associated 或 Function 有多少 N/A
print("AD_Associated 分布：")
print(df["AD_Associated"].value_counts(dropna=False))
