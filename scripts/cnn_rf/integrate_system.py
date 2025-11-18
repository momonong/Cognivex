"""
CNN-RF 系統整合腳本
Integration Script for CNN-RF System
"""

import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("CNN-RF System Integration")
print("="*80)

# 1. 檢查配置
print("\n[1] Checking Configuration...")
print("-"*80)

from scripts.cnn_rf.config import (
    DATA_ROOT, ROI_FEATURES_CSV, MODELS, MODEL_DIR,
    ATLAS_PATH, ATLAS_LABELS_PATH, MNI_TEMPLATE_PATH
)

checks = {
    "Data root": DATA_ROOT,
    "ROI features": ROI_FEATURES_CSV,
    "AAL3 atlas": ATLAS_PATH,
    "AAL3 labels": ATLAS_LABELS_PATH,
    "MNI template": MNI_TEMPLATE_PATH,
    "Model directory": MODEL_DIR
}

all_ok = True
for name, path in checks.items():
    exists = path.exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {name}: {path}")
    if not exists and name != "Model directory":
        all_ok = False

if not all_ok:
    print("\n[ERROR] Some required files are missing!")
    print("Please check the paths in scripts/cnn_rf/config.py")
    sys.exit(1)

# 2. 檢查模型
print("\n[2] Checking Models...")
print("-"*80)

for model_name, model_config in MODELS.items():
    model_path = model_config['path']
    exists = model_path.exists()
    status = "[OK]" if exists else "[NOT TRAINED]"
    print(f"  {status} {model_name}")
    print(f"       Path: {model_path}")
    print(f"       Classes: {model_config['classes']}")
    print(f"       Description: {model_config['description']}")

# 3. 測試推理接口
print("\n[3] Testing Inference Interface...")
print("-"*80)

try:
    from scripts.cnn_rf.inference import CNNRF_Predictor
    
    # 找到第一個存在的模型
    available_model = None
    for model_name, model_config in MODELS.items():
        if model_config['path'].exists():
            available_model = model_name
            break
    
    if available_model:
        print(f"  Testing with model: {available_model}")
        predictor = CNNRF_Predictor(
            model_path=str(MODELS[available_model]['path'])
        )
        print(f"  [OK] Predictor initialized successfully")
        print(f"       Classes: {predictor.class_names}")
    else:
        print(f"  [WARN] No trained models found")
        print(f"  [INFO] Run 'python scripts/cnn_rf/train_feat.py' to train models")
        
except Exception as e:
    print(f"  [ERROR] Failed to initialize predictor: {e}")
    import traceback
    traceback.print_exc()

# 4. 整合建議
print("\n[4] Integration Recommendations...")
print("-"*80)

print("""
To integrate CNN-RF into your system:

A. 與 Multimodal ROI 系統整合:
   1. 在 scripts/multimodal_roi/train.py 中導入:
      from scripts.cnn_rf.inference import CNNRF_Predictor
   
   2. 使用 CNN-RF 作為補充預測:
      cnn_rf = CNNRF_Predictor()
      results = cnn_rf.predict(features)

B. 與知識圖譜整合:
   1. 將重要腦區添加到 Neo4j:
      from scripts.cnn_rf.config import TOP_ROIS_NC_VS_AD
      # 創建 ROI 節點並標記重要性
   
   2. 連接 ROI 與疾病:
      # 創建 (ROI)-[:IMPORTANT_FOR]->(Disease) 關係

C. 創建統一預測接口:
   1. 創建 scripts/unified_predictor.py
   2. 整合多個模型的預測結果
   3. 提供統一的 API

D. Web 接口整合:
   1. 在 Flask/FastAPI 中添加 CNN-RF 端點
   2. 提供腦區可視化 API
   3. 返回預測結果和重要腦區
""")

# 5. 快速開始指南
print("\n[5] Quick Start Guide...")
print("-"*80)

print("""
Step 1: 訓練模型 (如果還沒有)
  python scripts/cnn_rf/train_feat.py

Step 2: 評估模型
  python scripts/cnn_rf/eval_feat.py

Step 3: 生成腦區可視化
  python scripts/cnn_rf/visualize_feat.py

Step 4: 使用推理接口
  python scripts/cnn_rf/inference.py

Step 5: 查看文檔
  cat scripts/cnn_rf/README.md
""")

# 6. 系統狀態總結
print("\n[6] System Status Summary...")
print("-"*80)

data_ok = DATA_ROOT.exists()
features_ok = ROI_FEATURES_CSV.exists()
atlas_ok = ATLAS_PATH.exists() and ATLAS_LABELS_PATH.exists()
models_ok = any(m['path'].exists() for m in MODELS.values())

print(f"  Data:     {'[OK]' if data_ok else '[MISSING]'}")
print(f"  Features: {'[OK]' if features_ok else '[MISSING]'}")
print(f"  Atlas:    {'[OK]' if atlas_ok else '[MISSING]'}")
print(f"  Models:   {'[OK]' if models_ok else '[NOT TRAINED]'}")

print("\n" + "="*80)

if data_ok and features_ok and atlas_ok and models_ok:
    print("[SUCCESS] CNN-RF system is fully integrated and ready to use!")
elif data_ok and features_ok and atlas_ok:
    print("[INFO] CNN-RF system is configured. Train models to complete setup.")
else:
    print("[WARN] CNN-RF system needs configuration. Check missing items above.")

print("="*80)
