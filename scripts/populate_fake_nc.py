import os
import random
import shutil

def populate_fake_nc(src_dir: str, dst_dir: str, num_samples: int = 20):
    os.makedirs(dst_dir, exist_ok=True)

    all_images = [f for f in os.listdir(src_dir) if f.endswith(".png")]
    if len(all_images) < num_samples:
        print(f"[WARNING] 來源圖片只有 {len(all_images)} 張，不足 {num_samples}，全部複製。")
        sample_images = all_images
    else:
        sample_images = random.sample(all_images, num_samples)

    for img in sample_images:
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(dst_dir, img)
        shutil.copy(src_path, dst_path)
        print(f"[COPY] {img} → NC")

if __name__ == "__main__":
    src = "data/images/AD"
    dst = "data/images/NC"
    num = 20  # 想複製幾張

    populate_fake_nc(src, dst, num)
    print("[DONE] NC 假資料建立完成")
