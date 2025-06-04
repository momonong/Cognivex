import os
import imageio
import re
from glob import glob
from tqdm import tqdm

def extract_slice_number(filename):
    match = re.search(r'slice(\d+)', filename)
    return int(match.group(1)) if match else -1

def extract_channel_number(filename):
    match = re.search(r'channel(\d+)', filename)
    return int(match.group(1)) if match else -1

def make_gifs(input_dir, output_dir, duration=0.2):
    os.makedirs(output_dir, exist_ok=True)
    
    # 找出所有圖片
    image_paths = glob(os.path.join(input_dir, '*.png'))
    
    # 根據 channel 分組
    channel_dict = {}
    for path in image_paths:
        channel = extract_channel_number(path)
        channel_dict.setdefault(channel, []).append(path)

    for channel, paths in tqdm(channel_dict.items(), desc="Generating GIFs"):
        # 根據 slice 編號排序
        sorted_paths = sorted(paths, key=extract_slice_number)

        images = []
        for img_path in sorted_paths:
            image = imageio.imread(img_path)
            images.append(image)

        gif_path = os.path.join(output_dir, f'channel{channel}_slices.gif')
        imageio.mimsave(gif_path, images, duration=duration)
        print(f'✅ Saved: {gif_path}')

if __name__ == "__main__":
    input_dir = "figures/capsnet/sub-14"
    output_dir = "figures/capsnet/gifs"
    make_gifs(input_dir, output_dir)
