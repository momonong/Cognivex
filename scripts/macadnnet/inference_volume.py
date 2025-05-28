import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from scripts.macadnnet.model import MCADNNet  # 載入你的模型

# ---------- 設定 ----------
csv_path = "data/slices_metadata_z30-60.csv"
output_dir = "output/macadnnet/conv_volume"
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------- 載入模型 ----------
model = MCADNNet(num_classes=2).to(device)
model.load_state_dict(torch.load("model/macadnnet/best_overall_model.pth", map_location=device))
model.eval()

# 擷取 conv2 activation
activation_store = {}
def hook_fn(module, input, output):
    activation_store["conv2"] = output.detach().cpu().numpy()

model.conv2.register_forward_hook(hook_fn)

# ---------- 載入 metadata ----------
df = pd.read_csv(csv_path)
subject_activations = defaultdict(list)

for _, row in tqdm(df.iterrows(), total=len(df)):
    subject_id = row["subject_id"]
    z = int(row["filename"].split("_z")[1].split("_")[0])
    t = int(row["filename"].split("_t")[1].split(".")[0])

    img = Image.open(row["path"]).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(img_tensor)
        act_map = activation_store["conv2"].squeeze(0)  # [C, H, W]
        avg_map = np.mean(act_map, axis=0)              # → [H, W]

    subject_activations[subject_id].append(((z, t), avg_map))

# ---------- 儲存每個 subject 的 conv 熱圖 volume ----------
meta_records = []
for subject_id, slices in subject_activations.items():
    slices.sort(key=lambda x: (x[0][0], x[0][1]))
    maps = [m for (_, m) in slices]
    volume = np.stack(maps, axis=0)  # shape: [num_slices, H, W]
    save_path = os.path.join(output_dir, f"{subject_id}_conv_volume.npy")
    np.save(save_path, volume)

    label = df[df["subject_id"] == subject_id]["label"].iloc[0]

    meta_records.append({
        "subject_id": subject_id,
        "label": label,
        "path": save_path,
        "num_slices": len(maps)
    })

# ---------- 儲存 meta CSV ----------
meta_df = pd.DataFrame(meta_records)
meta_df.to_csv(os.path.join(output_dir, "conv_volume_meta.csv"), index=False)
print("✅ 所有 conv activation volume 已儲存完成")
