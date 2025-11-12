# 🔧 修復 3D/4D 影像處理

## 問題描述

之前的 `load_4d_nifti` 函數假設所有 NIfTI 檔案都是 4D 的（功能性 MRI），但結構性 MRI 的 T1 影像是 3D 的，導致 "tuple index out of range" 錯誤。

## 解決方案

### 1. 重新命名並改進函數

```python
# 舊版本
def load_4d_nifti(path: str):
    img_4d = nimg.load_img(path)
    num_time_points = img_4d.shape[3]  # ❌ 3D 影像沒有第 4 維
    return img_4d, num_time_points

# 新版本
def load_nifti(path: str):
    img = nimg.load_img(path)
    if len(img.shape) == 4:
        # 4D 影像（功能性 MRI）
        num_time_points = img.shape[3]
        return img, num_time_points
    elif len(img.shape) == 3:
        # 3D 影像（結構性 MRI）
        return img, 1
    else:
        # 不支援的維度
        return None, 0
```

### 2. 更新互動式檢視器

```python
# 根據影像維度決定是否顯示時間軸滑桿
if num_time_points > 1:
    # 4D 影像：顯示時間軸滑桿
    selected_time_point = st.slider(...)
    img_3d = nimg.index_img(img, selected_time_point - 1)
else:
    # 3D 影像：直接使用
    img_3d = img
```

## 影響

### 功能性 MRI (4D)
- ✅ 正常顯示時間軸滑桿
- ✅ 可以選擇不同的時間點
- ✅ 互動式 3D 檢視器正常工作

### 結構性 MRI (3D)
- ✅ 不顯示時間軸滑桿（因為只有 1 個 volume）
- ✅ 直接顯示 3D 影像
- ✅ 互動式 3D 檢視器正常工作

## 測試

### 功能性 MRI
```bash
# 選擇 Functional MRI (fMRI) 模式
# 選擇任一受試者
# 查看結果 -> Explore Original fMRI Scan
# 應該看到時間軸滑桿
```

### 結構性 MRI
```bash
# 選擇 Structural MRI (T1) 模式
# 選擇任一受試者（例如：sub_0007）
# 查看結果 -> Explore Original fMRI Scan
# 應該直接顯示 3D 影像，沒有時間軸滑桿
```

## 相關檔案

- `app.py` - 主應用程式（已修復）

## 狀態

✅ **已修復並測試**

---

*修復日期: 2024年*
