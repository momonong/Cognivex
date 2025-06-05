import nibabel as nib
from nilearn.image import resample_to_img

resampled_act_img = resample_to_img(
    nib.load("output/capsnet/sub-14_conv3_strongest_masked.nii.gz"),
    nib.load("data/aal3/AAL3v1_1mm.nii.gz"),
    interpolation='nearest'
)

# 存起來看一下
resampled_path = "output/capsnet/sub-14_conv3_resampled_to_atlas.nii.gz"
resampled_act_img.to_filename(resampled_path)
