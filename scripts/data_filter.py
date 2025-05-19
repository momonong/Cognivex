import os
import pandas as pd
import argparse
import re

def extract_z_index(filename):
    """從檔名中解析 z 值，例如 AD_sub-07_z045_t003.png → 45"""
    match = re.search(r"_z(\d+)_t", filename)
    return int(match.group(1)) if match else -1

def filter_by_z_range(input_csv, output_csv, z_min=30, z_max=60):
    df = pd.read_csv(input_csv)

    # 解析 z 值
    df["z"] = df["path"].apply(lambda p: extract_z_index(os.path.basename(p)))

    # 篩選 z 範圍
    filtered_df = df[(df["z"] >= z_min) & (df["z"] <= z_max)].copy()
    filtered_df.drop(columns=["z"], inplace=True)

    # 儲存
    filtered_df.to_csv(output_csv, index=False)
    print(f"✅ 篩選完成：z ∈ [{z_min}, {z_max}]，共保留 {len(filtered_df)} 筆資料")
    print(f"📁 已儲存至：{output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="依 z-slice 範圍篩選切片 metadata")
    parser.add_argument("--input", type=str, default="data/slices_metadata.csv", help="原始 metadata CSV 路徑")
    parser.add_argument("--output", type=str, default="data/slices_metadata_z30-60.csv", help="篩選後儲存路徑")
    parser.add_argument("--zmin", type=int, default=30, help="最小 z 值")
    parser.add_argument("--zmax", type=int, default=60, help="最大 z 值")
    args = parser.parse_args()

    filter_by_z_range(args.input, args.output, args.zmin, args.zmax)
