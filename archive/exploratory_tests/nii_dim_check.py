import nibabel as nib

img = nib.load("data/raw/AD/sub-07/dswausub-027_S_6648_task-rest_bold.nii")
print("Shape:", img.shape)  # (91, 109, 91) 或 (91, 109, 91, T)
