import os
import sys
import glob
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom
from nilearn import plotting
import matplotlib.pyplot as plt

# =========== IG 歸因核心 ==============
def integrated_gradients(model, input_tensor, baseline=None, target_class=None, steps=50):
    device = input_tensor.device
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)
    input_tensor = input_tensor.to(device)
    baseline = baseline.to(device)

    # Collect scaled inputs
    scaled_inputs = [baseline + (float(i)/steps)*(input_tensor-baseline) for i in range(steps+1)]
    grads = []

    for x in scaled_inputs:
        x = x.clone().detach().requires_grad_(True)  # 這裡確保 x 是 leaf tensor
        model.zero_grad()
        out = model(x)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        score = out[0, target_class]
        grad = torch.autograd.grad(score, x)[0]  # 用 autograd.grad 而不是 .grad
        grads.append(grad.detach().clone())

    avg_grads = torch.stack(grads, dim=0).mean(dim=0)
    ig = (input_tensor-baseline) * avg_grads
    return ig.detach().cpu().squeeze().numpy()

def upsample_to_original(activation_map, original_shape):
    zoom_factor = np.array(original_shape) / np.array(activation_map.shape)
    upsampled = zoom(activation_map, zoom_factor, order=1)
    return upsampled

def save_as_nifti(np_map, ref_nifti_path, out_path):
    ref_img = nib.load(ref_nifti_path)
    affine = ref_img.affine
    header = ref_img.header
    assert np_map.shape == ref_img.shape[:3], f"Shape mismatch: {np_map.shape} vs {ref_img.shape}"
    act_img = nib.Nifti1Image(np_map.astype(np.float32), affine, header)
    nib.save(act_img, out_path)
    return out_path

def visualize_nilearn(bg_nii, act_nii, threshold=0.3, out_png=None):
    display = plotting.plot_stat_map(
        act_nii, bg_img=bg_nii, threshold=threshold,
        cmap='hot', display_mode='ortho', title="IG Attribution"
    )
    if out_png:
        display.savefig(out_png, dpi=120)
    plotting.show()

def show_top_regions(act_map_nifti, atlas_nii_path, atlas_label_path, threshold=0.3, top_n=5):
    from nilearn.image import resample_to_img
    import json
    act_img = nib.load(act_map_nifti)
    atlas_img = nib.load(atlas_nii_path)
    resampled_atlas = resample_to_img(atlas_img, act_img, interpolation='nearest')
    atlas_data = resampled_atlas.get_fdata()
    act_data = act_img.get_fdata()
    active_voxels = act_data > threshold
    active_regions = atlas_data[active_voxels]
    with open(atlas_label_path, 'r', encoding='utf-8') as jf:
        labels = json.load(jf)
    unique, counts = np.unique(active_regions, return_counts=True)
    regions = [(int(k), counts[i]) for i, k in enumerate(unique) if k > 0]
    regions = sorted(regions, key=lambda x: x[1], reverse=True)[:top_n]
    print('Top activated brain regions:')
    for idx, cnt in regions:
        print(f"- {labels[str(idx)]} ({cnt} voxels)")
    plotting.plot_roi(
        resampled_atlas, bg_img=act_map_nifti, title="Atlas Overlay", cmap='tab20'
    )
    plotting.show()

def run_ig_pipeline(
    model, nii_path, atlas_nii_path, atlas_label_path, output_dir="output", steps=50, threshold=0.3):
    os.makedirs(output_dir, exist_ok=True)
    img = nib.load(nii_path)
    img_data = img.get_fdata()
    shape = img_data.shape
    input_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1,1,D,H,W]
    ig_attr = integrated_gradients(model, input_tensor, steps=steps)
    if ig_attr.shape != shape:
        ig_attr = upsample_to_original(ig_attr, shape)
    act_nifti_path = os.path.join(output_dir, "ig_activation.nii.gz")
    save_as_nifti(ig_attr, nii_path, act_nifti_path)
    out_fig_path = os.path.join(output_dir, "overlay.png")
    visualize_nilearn(nii_path, act_nifti_path, threshold, out_fig_path)
    show_top_regions(act_nifti_path, atlas_nii_path, atlas_label_path, threshold)
    print("\nAll outputs saved in", output_dir)
    return act_nifti_path, out_fig_path

# ========== 主要批次執行 ===========
if __name__ == "__main__":
    sys.path.append('D:/projects/Cognivex')
    from app.core.cnn_3d.model import Simple3DCNN_InstanceNorm

    # ======= 基本路徑參數 =======
    MODEL_WEIGHT_DIR = "model/cnn_3d"
    model_weight_paths = sorted(glob.glob(os.path.join(MODEL_WEIGHT_DIR, "cnn_3d_fold_*.pth")))

    nii_file = r"E:/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii"

    ATLAS_NII_PATH = "data/aal3/AAL3v1_1mm.nii.gz"
    ATLAS_LABEL_PATH = "data/aal3/AAL3v1.json"
    IG_STEPS = 50
    IG_THRESHOLD = 0.3

    for idx, weight_path in enumerate(model_weight_paths, 1):
        print(f"\n=== Fold {idx} : {weight_path} ===")
        model = Simple3DCNN_InstanceNorm()
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        model.eval()
        output_dir = f"output_demo/fold_{idx}"
        run_ig_pipeline(
            model=model,
            nii_path=nii_file,
            atlas_nii_path=ATLAS_NII_PATH,
            atlas_label_path=ATLAS_LABEL_PATH,
            output_dir=output_dir,
            steps=IG_STEPS,
            threshold=IG_THRESHOLD
        )
    print("\n全部 fold 執行完畢，結果請至 output_demo目錄查看每個 fold 的結果！")
