import csv

input_csv = "output/region_to_network_mapping.csv"

network_id_to_name = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attention",
    4: "Ventral Attention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default Mode"
}

output_rows = []

with open(input_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        region_id = int(row["aal_id"])
        region_name = row["region_name"]
        network_id = int(row["yeo7_network_id"])

        # 對應不到 network 就標記為 Unknown
        network_name = network_id_to_name.get(network_id, "Unknown")

        output_rows.append({
            "region_id": region_id,
            "region_name": region_name,
            "network_id": network_id,
            "network_name": network_name
        })

# 儲存為新的含 network 名稱的 CSV
with open("output/region_to_network_mapping_with_names.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["region_id", "region_name", "network_id", "network_name"])
    writer.writeheader()
    writer.writerows(output_rows)

print("✅ 已產生包含 network 名稱的新 CSV：output/region_to_network_mapping_with_names.csv")
