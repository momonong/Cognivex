import os
from nilearn import plotting
import matplotlib.pyplot as plt

# --- 1. 設定 ---
# 你的 MNI 模板 (底圖)
TEMPLATE_PATH = r"data/templates/MNI152_T1_1mm_brain.nii.gz"
# 你的 ROI 特徵圖 (疊加圖)
ROI_MAP_PATH = r"output/cnn_rf/NC_vs_AD_top_features_map.nii.gz"
# 最終輸出的 PNG 圖片
OUTPUT_PNG = r"output/cnn_rf/NC_vs_AD_top_features_visualization.png"

def plot_feature_map():
    print(f"[*] 正在載入底圖: {TEMPLATE_PATH}")
    print(f"[*] 正在載入疊加圖: {ROI_MAP_PATH}")
    
    # 檢查檔案是否存在
    if not os.path.exists(TEMPLATE_PATH):
        print(f"[!] 錯誤: 找不到 MNI 模板 {TEMPLATE_PATH}")
        return
    if not os.path.exists(ROI_MAP_PATH):
        print(f"[!] 錯誤: 找不到 ROI 特徵圖 {ROI_MAP_PATH}")
        print("[!] 請先執行 'python -m src.visualize_features'")
        return

    # --- 2. 執行繪圖 ---
    print(f"[*] 正在繪製 PNG 圖片...")
    try:
        # 使用 nilearn.plotting.plot_roi
        # bg_img: 指定灰色的 MNI 大腦底圖
        # roi_img: 指定你的紅色腦區疊加圖
        # display_mode='ortho': 顯示三個切面 (矢狀、冠狀、軸狀)
        # cut_coords: 自動找到紅色腦區的中心點
        # draw_cross=False: 隱藏藍色十字線
        # output_file: 將結果儲存到檔案
        plotting.plot_roi(
            roi_img=ROI_MAP_PATH,
            bg_img=TEMPLATE_PATH,
            display_mode='mosaic',
            draw_cross=False,
            annotate=True,
            cmap='autumn',         # 使用 'autumn' (紅黃) 色彩映射
            output_file=OUTPUT_PNG
        )
        
        # 關閉 matplotlib 的彈出視窗
        plt.close()
        
        print(f"\n[SUCCESS] 成功儲存視覺化 PNG！")
        print(f"  -> 檔案儲存於: {OUTPUT_PNG}")
        
    except Exception as e:
        print(f"[!] 繪圖時發生錯誤: {e}")
        print("[!] 請確認 'pip install nilearn matplotlib' 已成功執行。")

if __name__ == "__main__":
    plot_feature_map()