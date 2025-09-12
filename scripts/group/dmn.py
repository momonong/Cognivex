from nilearn import plotting, datasets, image
import numpy as np
from nilearn.image import new_img_like

yeo7 = datasets.fetch_atlas_yeo_2011()['thin_7']
mni = datasets.load_mni152_template()
dmn_mask = image.math_img('img == 7', img=yeo7)
dmn_mask = new_img_like(dmn_mask, np.squeeze(dmn_mask.get_fdata()))

# 先畫好 ROI 主色塊
display = plotting.plot_roi(
    dmn_mask,
    bg_img=mni,
    title="Yeo7 DMN (with outline)",
    cmap='tab10',
    alpha=0.7,
    black_bg=False,
    display_mode="mosaic",
)

# 疊加外框輪廓線
display.add_contours(
    dmn_mask,
    levels=[0.5],           # 0/1 mask，通常設 0.5
    colors='b',           # 外框顏色
    linewidths=2,
    alpha=0.3               # 外框透明度
)

display.savefig('figures/masks/dmn_outline.png')
