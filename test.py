# st_viewer_test.py (TypeError Fix)
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

# --- 視覺化相關 ---
from nilearn import plotting
from nilearn import image as nimg

# --- 配置 ---
NIFTI_FILE_PATH = "data/raw/CN/sub-01/dswausub-009_S_0751_task-rest_bold.nii.gz"

# --- 快取函式 (保持不變) ---
@st.cache_resource(show_spinner="正在載入並處理 NIfTI 檔案...")
def load_4d_nifti(path: str):
    try:
        img_4d = nimg.load_img(path)
        num_time_points = img_4d.shape[3]
        return img_4d, num_time_points
    except Exception as e:
        st.error(f"載入或處理 4D 檔案失敗: {path}. 錯誤: {e}")
        return None, 0

# --- STREAMLIT 前端介面 ---
st.set_page_config(page_title="4D NIfTI Viewer Test", layout="wide")
st.title("🔬 4D NIfTI 互動式檢視器 (帶時間軸)")
st.markdown("拖動下方的**時間點**滑桿，探索 fMRI 數據隨時間的變化。")

if not Path(NIFTI_FILE_PATH).exists():
    st.error(f"測試檔案不存在，請檢查 `NIFTI_FILE_PATH` 的設定: {NIFTI_FILE_PATH}")
else:
    img_4d, num_time_points = load_4d_nifti(NIFTI_FILE_PATH)
    
    if img_4d and num_time_points > 0:
        
        # ---### 關鍵修改之處 ###---
        # Explicitly cast all numeric arguments to Python's int type
        # to prevent any type mismatches.
        selected_time_point = st.slider(
            'Time Point (Volume)', 
            min_value=int(0), 
            max_value=int(num_time_points - 1), 
            value=int(0),
            help=f"這個 fMRI 掃描共有 {num_time_points} 個時間點 (volumes)。"
        )

        st.header(f"互動式三視圖檢視器 (時間點: {selected_time_point})")
        
        img_3d_at_t = nimg.index_img(img_4d, selected_time_point)

        viewer = plotting.view_img(
            img_3d_at_t, 
            bg_img=None, 
            cmap='gray', 
            threshold=None, 
            title=f"Volume at T={selected_time_point}",
            resampling_interpolation='nearest'
        )
        
        components.html(viewer.html, height=600, scrolling=False)