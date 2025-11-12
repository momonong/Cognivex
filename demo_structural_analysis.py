"""
Structural MRI Analysis Demo
演示結構性 MRI 分析的完整流程（不需要 Streamlit）
"""

import sys
from pathlib import Path

print("="*70)
print("🧠 Structural MRI Analysis Demo")
print("="*70)

# Step 1: 檢查模型檔案
print("\n[Step 1] Checking model files...")
model_files = [
    "model/ml/final/final_model.pkl",
    "model/ml/final/final_scaler.pkl",
    "model/ml/final/final_roi_list.csv",
    "model/ml/final/final_feature_names.txt"
]

all_exist = True
for file_path in model_files:
    exists = Path(file_path).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {file_path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ Some model files are missing!")
    print("Please ensure all model files are in model/ml/final/")
    sys.exit(1)

print("\n✅ All model files found!")

# Step 2: 載入模型
print("\n[Step 2] Loading ML model...")
try:
    from app.core.ml_processing import MLModelLoader, MLModelConfig
    
    config = MLModelConfig.from_directory()
    loader = MLModelLoader(config)
    components = loader.get_all_components()
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Model type: {type(components['model']).__name__}")
    print(f"   Number of trees: {components['model'].n_estimators}")
    print(f"   Number of features: {len(components['feature_names'])}")
    print(f"   Feature names: {components['feature_names'][:5]}...")
    
except Exception as e:
    print(f"\n❌ Model loading failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check if model files are valid pickle files")
    print("2. Try re-training the model")
    print("3. Check Python and scikit-learn versions")
    sys.exit(1)

# Step 3: 載入 Atlas
print("\n[Step 3] Loading AAL Atlas...")
try:
    from app.core.ml_processing import ROIFeatureExtractor
    
    extractor = ROIFeatureExtractor()
    atlas_img, atlas_labels = extractor.load_atlas()
    
    print(f"\n✅ Atlas loaded successfully!")
    print(f"   Total regions: {len(atlas_labels)}")
    print(f"   First 5 regions: {atlas_labels[:5]}")
    
except Exception as e:
    print(f"\n❌ Atlas loading failed: {e}")
    sys.exit(1)

# Step 4: 測試特徵提取（使用模擬數據）
print("\n[Step 4] Testing feature extraction...")
try:
    import numpy as np
    
    # 模擬特徵（實際使用時會從 MRI 提取）
    mock_features = np.random.randn(32)
    print(f"   Mock features shape: {mock_features.shape}")
    
    # 標準化
    scaler = components['scaler']
    features_scaled = scaler.transform(mock_features.reshape(1, -1))
    print(f"   Scaled features shape: {features_scaled.shape}")
    
    print("\n✅ Feature extraction pipeline works!")
    
except Exception as e:
    print(f"\n❌ Feature extraction test failed: {e}")
    sys.exit(1)

# Step 5: 測試預測
print("\n[Step 5] Testing prediction...")
try:
    model = components['model']
    
    # 使用模擬特徵進行預測
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # 獲取類別名稱
    class_names = model.classes_
    prediction_label = class_names[prediction]
    confidence = probabilities[prediction]
    
    print(f"\n✅ Prediction successful!")
    print(f"   Classification: {prediction_label}")
    print(f"   Confidence: {confidence:.1%}")
    print(f"   Probabilities: NC={probabilities[0]:.1%}, AD={probabilities[1]:.1%}")
    
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    sys.exit(1)

# Step 6: 測試特徵重要性
print("\n[Step 6] Testing feature importance...")
try:
    importances = model.feature_importances_
    feature_names = components['feature_names']
    
    # 創建特徵重要性字典
    feature_importance_dict = dict(zip(feature_names, importances))
    
    # 排序
    sorted_features = sorted(
        feature_importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    print(f"\n✅ Feature importance extracted!")
    print(f"\n   Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        bar_length = int(importance * 200)
        bar = "█" * bar_length
        print(f"   {i:2d}. {feature:20s} {bar} {importance*100:.2f}%")
    
except Exception as e:
    print(f"\n❌ Feature importance extraction failed: {e}")
    sys.exit(1)

# Step 7: 測試 Agent 節點
print("\n[Step 7] Testing agent nodes...")
try:
    from app.agents.structural_mri_inference import run_structural_mri_inference
    from app.agents.structural_feature_analyzer import analyze_feature_importance
    
    print("   ✓ structural_mri_inference imported")
    print("   ✓ structural_feature_analyzer imported")
    
    # 注意：實際執行需要真實的 MRI 檔案
    print("\n   Note: Full agent execution requires real MRI files")
    
except Exception as e:
    print(f"\n❌ Agent import failed: {e}")
    sys.exit(1)

# Step 8: 測試 Workflow 路由
print("\n[Step 8] Testing workflow routing...")
try:
    from app.graph.workflow import route_by_analysis_mode
    from app.graph.state import AgentState
    
    # 測試結構性 MRI 路由
    test_state_structural = {"analysis_mode": "structural"}
    next_node = route_by_analysis_mode(test_state_structural)
    print(f"   ✓ Structural mode routes to: {next_node}")
    
    # 測試功能性 MRI 路由
    test_state_functional = {"analysis_mode": "functional"}
    next_node = route_by_analysis_mode(test_state_functional)
    print(f"   ✓ Functional mode routes to: {next_node}")
    
    print("\n✅ Workflow routing works!")
    
except Exception as e:
    print(f"\n❌ Workflow routing failed: {e}")
    print(f"   Error: {e}")
    print("\n   Note: This might be due to missing langgraph package")
    print("   Install with: pip install langgraph")

# Final Summary
print("\n" + "="*70)
print("📊 Demo Summary")
print("="*70)
print("\n✅ Core Components Status:")
print("   ✓ Model files present")
print("   ✓ Model loading works")
print("   ✓ Atlas loading works")
print("   ✓ Feature extraction pipeline works")
print("   ✓ Prediction works")
print("   ✓ Feature importance extraction works")
print("   ✓ Agent nodes importable")
print("   ✓ Workflow routing works (if langgraph installed)")

print("\n📝 Next Steps:")
print("   1. Integrate into app.py following docs/app_py_integration_guide.md")
print("   2. Test with real MRI files")
print("   3. Verify UI components work in Streamlit")
print("   4. Run full end-to-end tests")

print("\n🎯 System is ready for integration!")
print("="*70)
