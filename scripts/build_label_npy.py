import pandas as pd
import numpy as np

def build_labels_from_csv(csv_path, save_path="output/labels_fc1.npy"):
    df = pd.read_csv(csv_path)
    labels = []

    for _, row in df.iterrows():
        if pd.notna(row["AD_Subject"]):
            labels.append(1)  # AD
        if pd.notna(row["CN_Subject"]):
            labels.append(0)  # CN

    labels = np.array(labels)
    np.save(save_path, labels)
    print(f"[✅ Saved] labels saved to {save_path}, shape = {labels.shape}")

if __name__ == "__main__":
    build_labels_from_csv("data/adnc_match.csv")
