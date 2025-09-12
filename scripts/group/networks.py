from nilearn import plotting, datasets

# 下載 Yeo7 網路的 atlas（NIfTI 格式）
yeo7 = datasets.fetch_atlas_yeo_2011()['thin_7']

# 可選：載入標準腦
from nilearn import image
mni = datasets.load_mni152_template()
# 如果你的數據有自定標準腦，可以用 image.load_img('你的標準腦路徑.nii.gz')

# 畫色塊示意圖（腦圖疊色）
display = plotting.plot_roi(
    yeo7,
    bg_img=mni,
    title="Yeo7 Networks",
    cmap='tab10',     # 10 個鮮豔對比色
    alpha=1,        # 色塊的透明度（調高會更明顯）
    draw_cross=False,
    display_mode="mosaic",
)
output_path = "figures/masks/networks.png"
display.savefig(output_path)