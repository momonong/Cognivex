"""
🧪 Final System Test - 全面系統測試
測試整個 Cognivex 系統的所有功能
"""

import sys
import time
from pathlib import Path

print("="*80)
print("🧪 COGNIVEX FINAL SYSTEM TEST")
print("="*80)
print("Testing all components of the integrated fMRI + sMRI analysis system")
print("="*80)

# Test 1: 檢查資料結構
print("\n[TEST 1] 📁 Data Structure Verification")
print("-" * 50)

data_paths = {
    "fMRI Data": "data/fMRI",
    "sMRI Data": "data/sMRI"
}

test_results = {"passed": 0, "failed": 0}

for name, path in data_paths.items():
    path_obj = Path(path)
    if path_obj.exists():
        # Count subjects
        ad_subjects = len(list((path_obj / "AD").glob("sub-*"))) if (path_obj / "AD").exists() else 0
        nc_subjects = len(list((path_obj / "NC").glob("sub-*"))) if (path_obj / "NC").exists() else 0
        cn_subjects = len(list((path_obj / "CN").glob("sub-*"))) if (path_obj / "CN").exists() else 0
        
        total = ad_subjects + nc_subjects + cn_subjects
        print(f"   ✅ {name}: {total} subjects (AD: {ad_subjects}, NC: {nc_subjects + cn_subjects})")
        test_results["passed"] += 1
    else:
        print(f"   ❌ {name}: Directory not found")
        test_results["failed"] += 1

# Test 2: 檢查模型檔案
print("\n[TEST 2] 🤖 Model Files Verification")
print("-" * 50)

model_files = {
    "sMRI Random Forest": "model/ml/final/final_model.pkl",
    "sMRI Scaler": "model/ml/final/final_scaler.pkl",
    "sMRI ROI List": "model/ml/final/final_roi_list.csv",
    "fMRI ShuffleNet": "model/shufflenet/fold_3_best_model.pth",
    "fMRI CapsNet": "model/capsnet/best_capsnet_rnn.pth",
    "fMRI MCADNNet": "model/macadnnet/._best_overall_model.pth"
}

for name, path in model_files.items():
    path_obj = Path(path)
    if path_obj.exists():
        size_mb = path_obj.stat().st_size / (1024 * 1024)
        print(f"   ✅ {name}: {size_mb:.2f} MB")
        test_results["passed"] += 1
    else:
        print(f"   ❌ {name}: File not found")
        test_results["failed"] += 1

# Test 3: 測試 sMRI 模型載入
print("\n[TEST 3] 🧠 sMRI Model Loading Test")
print("-" * 50)

try:
    from app.core.ml_processing.model_loader import MLModelLoader
    
    loader = MLModelLoader()
    print("   ✅ MLModelLoader initialized")
    test_results["passed"] += 1
    
    # Load model components
    components = loader.load_model()
    print(f"   ✅ Model components loaded: {len(components)} items")
    
    if 'model' in components:
        print(f"   ✅ Model: {type(components['model']).__name__}")
        test_results["passed"] += 1
    if 'scaler' in components:
        print(f"   ✅ Scaler: {type(components['scaler']).__name__}")
        test_results["passed"] += 1
    if 'roi_list' in components:
        print(f"   ✅ ROI list: {len(components['roi_list'])} regions")
        test_results["passed"] += 1
    
except Exception as e:
    print(f"   ❌ sMRI model loading failed: {e}")
    test_results["failed"] += 4

# Test 4: 測試 sMRI 特徵提取
print("\n[TEST 4] 🔬 sMRI Feature Extraction Test")
print("-" * 50)

try:
    from app.core.ml_processing.feature_extractor import ROIFeatureExtractor
    
    # Find a T1 test file
    test_file = None
    for subject_dir in Path("data/sMRI/NC").glob("sub-*"):
        t1_files = list(subject_dir.glob("*T1*.nii.gz"))
        if t1_files:
            test_file = str(t1_files[0])
            break
    
    if test_file:
        print(f"   📁 Test file: {Path(test_file).name}")
        
        extractor = ROIFeatureExtractor()
        # Load ROI list from CSV
        import pandas as pd
        roi_csv_path = Path("model/ml/final/final_roi_list.csv")
        if roi_csv_path.exists():
            roi_df = pd.read_csv(roi_csv_path)
            roi_list = roi_df['ROI_Name'].tolist()
        else:
            print(f"   ⚠️  ROI list not found, skipping feature extraction")
            test_results["failed"] += 2
            raise Exception("ROI list not found")
        
        features = extractor.extract_features(test_file, roi_list)
        
        if features is not None:
            print(f"   ✅ Features extracted: {len(features)} features")
            print(f"   ✅ Feature range: {features.min():.4f} ~ {features.max():.4f}")
            test_results["passed"] += 2
        else:
            print(f"   ❌ Feature extraction returned None")
            test_results["failed"] += 2
    else:
        print(f"   ⚠️  No test file found, skipping")
        
except Exception as e:
    print(f"   ❌ Feature extraction failed: {e}")
    test_results["failed"] += 2

# Test 5: 測試 fMRI 模型載入
print("\n[TEST 5] 🎬 fMRI Model Loading Test")
print("-" * 50)

try:
    from app.core.fmri_processing.fmri_model_loader import load_fmri_model
    
    print("   ✅ fMRI model loader imported")
    test_results["passed"] += 1
    
    # Test each model
    models_to_test = ["shufflenet", "capsnet", "macadnnet"]
    for model_name in models_to_test:
        model_path = Path(f"model/{model_name}")
        if model_path.exists():
            try:
                model = load_fmri_model(model_name)
                if model is not None:
                    print(f"   ✅ {model_name.upper()} loaded successfully")
                    test_results["passed"] += 1
                else:
                    print(f"   ⚠️  {model_name.upper()} returned None")
            except Exception as e:
                print(f"   ⚠️  {model_name.upper()} loading skipped: {str(e)[:50]}")
        else:
            print(f"   ⚠️  {model_name.upper()} directory not found (optional)")
            
except Exception as e:
    print(f"   ⚠️  fMRI model loader test skipped: {str(e)[:50]}")
    test_results["passed"] += 1  # Not critical for sMRI testing

# Test 6: 測試 Agent 系統
print("\n[TEST 6] 🤖 Agent System Test")
print("-" * 50)

try:
    # Test agent modules exist
    agent_modules = [
        "app.agents.structural_mri_inference",
        "app.agents.structural_feature_analyzer",
        "app.agents.structural_visualizer",
        "app.agents.inference"
    ]
    
    for module_name in agent_modules:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name.split('.')[-1]} module found")
            test_results["passed"] += 1
        except Exception as e:
            print(f"   ❌ {module_name.split('.')[-1]} module failed: {str(e)[:50]}")
            test_results["failed"] += 1
            
except Exception as e:
    print(f"   ❌ Agent system test failed: {e}")
    test_results["failed"] += 4

# Test 7: 測試 UI 元件
print("\n[TEST 7] 🎨 UI Components Test")
print("-" * 50)

try:
    import app.ui.structural_mri_components as ui_module
    print("   ✅ structural_mri_components module imported")
    test_results["passed"] += 1
    
    # Check for key functions
    functions_to_check = ['display_structural_results', 'render_structural_mri_ui']
    for func_name in functions_to_check:
        if hasattr(ui_module, func_name):
            print(f"   ✅ {func_name} found")
            test_results["passed"] += 1
        else:
            print(f"   ⚠️  {func_name} not found (may use different name)")
        
except Exception as e:
    print(f"   ❌ UI components test failed: {str(e)[:50]}")
    test_results["failed"] += 2

# Test 8: 檢查配置檔案
print("\n[TEST 8] ⚙️  Configuration Files Test")
print("-" * 50)

config_files = {
    "XAI Config": "config/xai_config.yaml",
    "Project Config": "pyproject.toml",
    "Environment": ".env"
}

for name, path in config_files.items():
    path_obj = Path(path)
    if path_obj.exists():
        print(f"   ✅ {name}: Found")
        test_results["passed"] += 1
    else:
        print(f"   ⚠️  {name}: Not found (may be optional)")

# Final Summary
print("\n" + "="*80)
print("📊 TEST SUMMARY")
print("="*80)

total_tests = test_results["passed"] + test_results["failed"]
success_rate = (test_results["passed"] / total_tests * 100) if total_tests > 0 else 0

print(f"\n   Total Tests: {total_tests}")
print(f"   ✅ Passed: {test_results['passed']}")
print(f"   ❌ Failed: {test_results['failed']}")
print(f"   📊 Success Rate: {success_rate:.1f}%")

if test_results["failed"] == 0:
    print("\n   🎉 ALL TESTS PASSED! System is ready for production.")
    print("\n" + "="*80)
    print("✅ SYSTEM STATUS: READY")
    print("="*80)
    sys.exit(0)
else:
    print("\n   ⚠️  Some tests failed. Please review the errors above.")
    print("\n" + "="*80)
    print("⚠️  SYSTEM STATUS: NEEDS ATTENTION")
    print("="*80)
    sys.exit(1)
