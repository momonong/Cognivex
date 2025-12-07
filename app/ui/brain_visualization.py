"""
Brain Region Visualization Component
使用 nilearn 生成腦區視覺化圖片
"""

import os
import shutil
import streamlit as st
from pathlib import Path
from nilearn import plotting
import matplotlib.pyplot as plt


def generate_brain_visualization(subject_id: str, output_dir: str = "output/visualizations"):
    """
    生成腦區視覺化圖片並複製 NIfTI 檔案
    
    Args:
        subject_id: 受試者 ID
        output_dir: 輸出目錄
        
    Returns:
        tuple: (圖片路徑, NIfTI 檔案路徑)，如果失敗則返回 (None, None)
    """
    # 設定路徑
    TEMPLATE_PATH = "data/templates/MNI152_T1_1mm_brain.nii.gz"
    ROI_MAP_PATH = "output/cnn_rf/NC_vs_AD_top_features_map.nii.gz"
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 輸出檔案名稱
    output_png = os.path.join(output_dir, f"{subject_id}_brain_visualization.png")
    output_nii = os.path.join(output_dir, f"{subject_id}_top_features_map.nii.gz")
    
    # 檢查必要檔案是否存在
    if not os.path.exists(TEMPLATE_PATH):
        st.warning(f"找不到 MNI 模板: {TEMPLATE_PATH}")
        return None, None
    
    if not os.path.exists(ROI_MAP_PATH):
        st.warning(f"找不到 ROI 特徵圖: {ROI_MAP_PATH}")
        return None, None
    
    try:
        # 複製 NIfTI 檔案到輸出目錄
        shutil.copy2(ROI_MAP_PATH, output_nii)
        
        # 使用 nilearn 生成視覺化
        plotting.plot_roi(
            roi_img=ROI_MAP_PATH,
            bg_img=TEMPLATE_PATH,
            display_mode='mosaic',
            draw_cross=False,
            annotate=True,
            cmap='autumn',  # 紅黃色彩映射
            output_file=output_png
        )
        
        # 關閉 matplotlib 視窗
        plt.close()
        
        return output_png, output_nii
        
    except Exception as e:
        st.error(f"生成腦區視覺化時發生錯誤: {e}")
        return None, None


def render_brain_visualization(subject_id: str):
    """
    在 Streamlit 中顯示腦區視覺化
    
    Args:
        subject_id: 受試者 ID
    """
    st.markdown("### Brain Region Visualization")
    st.markdown("Model-identified important brain regions overlaid on MNI template")
    
    # 生成視覺化圖片和 NIfTI 檔案
    with st.spinner("Generating brain visualization..."):
        image_path, nii_path = generate_brain_visualization(subject_id)
    
    if image_path and os.path.exists(image_path):
        # 讀取圖片為 bytes 來避免 Streamlit 快取問題
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        
        # 顯示圖片
        st.image(image_bytes, width='stretch')
        
        # 提供 NIfTI 檔案下載
        if nii_path and os.path.exists(nii_path):
            with open(nii_path, "rb") as file:
                st.download_button(
                    label="Download NIfTI File (.nii.gz)",
                    data=file,
                    file_name=f"{subject_id}_top_features_map.nii.gz",
                    mime="application/gzip",
                    help="Download the brain region feature map in NIfTI format for further analysis"
                )
    else:
        st.info("Brain visualization not available. Please ensure ROI feature map exists.")
