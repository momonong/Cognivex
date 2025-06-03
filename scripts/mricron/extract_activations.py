# scripts/activation_extraction.py

import os
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from scripts.macadnnet.model import MCADNNet  # 自訂模型

# ---------- 設定 ----------
csv_path = "data/activation_slices_metadata.csv"  # 來自第一步的 metadata
weights_path = "model/macadnnet/best_overall_model.pth"
output_dir = "output/activations"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------- 載入模型 ----------
model = MCADNNet(num_classes=2).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# 抽取 conv2 activation
activation_store = {}
def hook_fn(module, input, output):
    activation_store["conv2"] = output.detach().cpu().numpy()

model.conv2.register_forward_hook(hook_fn)

# ---------- 載入 metadata ----------
df = pd.read_csv(csv_path)
df = df[df["subject_id"] == "sub-14"]  # * 僅處理 sub-14 的資料。注意：這行是為了測試，實際使用時可移除或修改
subject_activations = defaultdict(list)

for _, row in tqdm(df.iterrows(), total=len(df)):
    subject_id = row["subject_id"]
    z = int(row["z"])
    t = int(row["t"])
    image_path = row["path"]

    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(img_tensor)
        conv_output = activation_store["conv2"].squeeze(0)  # shape: [C, H, W]
        avg_map = np.mean(conv_output, axis=0)              # shape: [H, W]

    subject_activations[subject_id].append(((z, t), avg_map))

# ---------- 儲存每個 subject 的 conv activation volume ----------
meta_records = []
for subject_id, slices in subject_activations.items():
    slices.sort(key=lambda x: (x[0][0], x[0][1]))  # 依照 (z, t) 排序
    volume = np.stack([m for (_, m) in slices], axis=0)  # shape: [num_slices, H, W]
    save_path = os.path.join(output_dir, f"{subject_id}_conv_volume.npy")
    np.save(save_path, volume)

    label = df[df["subject_id"] == subject_id]["label"].iloc[0]
    meta_records.append({
        "subject_id": subject_id,
        "label": label,
        "path": save_path,
        "num_slices": len(slices)
    })

# ---------- 儲存 metadata ----------
meta_df = pd.DataFrame(meta_records)
meta_df.to_csv(os.path.join(output_dir, "activation_volume_meta.csv"), index=False)
print("✅ 所有 conv activation volume 已完成並儲存")
