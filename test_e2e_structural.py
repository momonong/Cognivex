"""
End-to-End Test for Structural MRI Analysis
完整的端到端測試，模擬真實使用場景
"""

import sys
import glob
from pathlib import Path

print("="*70)
print("🧠 End-to-End Structural MRI Analysis Test")
print("="*70)

# Step 1: 尋找測試用的 MRI 檔案
print("\n[Step 1] Finding test MRI files...")
mri_files = glob.glob("data/raw/*/sub-*/*.nii.gz")

if not mri_files:
    print("❌ No MRI files found in data/raw/")
    print("Please ensure you have MRI data in the correct location")
    sys.exit(1)

# 選擇第一個檔案作為測試
test_file = mri_files[0]
print(f"✓ Found {len(mri_files)} MRI files")
print(f"✓ Using test file: {test_file}")

# 從路徑提取 subject_id 和 ground_truth
parts = test_file.split('/')
if len(parts) >= 3:
    ground_truth = parts[-3]  # AD or NC
    subject_id = parts[-2]     # sub-XXX
else:
    ground_truth = "Unknown"
    subject_id = "test_subject"

print(f"✓ Subject ID: {subject_id}")
print(f"✓ Ground Truth: {ground_truth}")

# Step 2: 準備初始狀態
print("\n[Step 2] Preparing initial state...")
initial_state = {
    "subject_id": subject_id,
    "fmri_scan_path": test_file,
    "analysis_mode": "structural",
    "trace_log": [],
    "error_log": []
}
print("✓ Initial state prepared")

# Step 3: 執行 Workflow
print("\n[Step 3] Running workflow...")
print("This may take 10-30 seconds...")

try:
    from app.graph.workflow import app
    
    print("\n" + "-"*70)
    final_state = app.invoke(initial_state)
    print("-"*70)
    
    print("\n✅ Workflow completed successfully!")
    
except Exception as e:
    print(f"\n❌ Workflow failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: 檢查結果
print("\n[Step 4] Checking results...")

# 檢查必要的輸出
required_fields = [
    "classification_result",
    "prediction_confidence",
    "roi_features",
    "feature_importances",
    "activated_regions",
    "visualization_paths"
]

missing_fields = []
for field in required_fields:
    if field not in final_state or final_state[field] is None:
        missing_fields.append(field)
        print(f"  ⚠️  Missing: {field}")
    else:
        print(f"  ✓ Present: {field}")

if missing_fields:
    print(f"\n⚠️  Warning: {len(missing_fields)} fields are missing")
else:
    print("\n✅ All required fields present!")

# Step 5: 顯示結果摘要
print("\n" + "="*70)
print("📊 Analysis Results Summary")
print("="*70)

# 基本資訊
print(f"\n📋 Subject Information:")
print(f"   Subject ID: {subject_id}")
print(f"   Ground Truth: {ground_truth}")
print(f"   MRI File: {Path(test_file).name}")

# 預測結果
classification = final_state.get("classification_result", "N/A")
confidence = final_state.get("prediction_confidence", 0)

print(f"\n🎯 Prediction Results:")
print(f"   Classification: {classification}")
print(f"   Confidence: {confidence:.1%}" if confidence else "   Confidence: N/A")

# 驗證結果
if ground_truth != "Unknown" and classification != "N/A":
    if ground_truth == classification:
        print(f"   Status: ✅ CORRECT (matches ground truth)")
    else:
        print(f"   Status: ❌ INCORRECT (ground truth: {ground_truth})")

# 特徵重要性
feature_importances = final_state.get("feature_importances", {})
if feature_importances:
    print(f"\n📊 Top 5 Important Features:")
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    for i, (feature, importance) in enumerate(sorted_features, 1):
        bar_length = int(importance * 100)
        bar = "█" * bar_length
        print(f"   {i}. {feature:20s} {bar} {importance*100:.2f}%")

# 視覺化檔案
viz_paths = final_state.get("visualization_paths", [])
if viz_paths:
    print(f"\n🖼️  Visualization Files:")
    for path in viz_paths:
        exists = "✓" if Path(path).exists() else "✗"
        print(f"   {exists} {path}")

# 報告
reports = final_state.get("generated_reports", {})
if reports:
    print(f"\n📄 Generated Reports:")
    if "en" in reports:
        en_length = len(reports["en"])
        print(f"   ✓ English report ({en_length} characters)")
    if "zh" in reports:
        zh_length = len(reports["zh"])
        print(f"   ✓ Chinese report ({zh_length} characters)")

# 錯誤日誌
error_log = final_state.get("error_log", [])
if error_log:
    print(f"\n⚠️  Errors encountered:")
    for error in error_log:
        print(f"   - {error}")
else:
    print(f"\n✅ No errors encountered")

# 追蹤日誌
trace_log = final_state.get("trace_log", [])
if trace_log:
    print(f"\n📝 Processing Steps ({len(trace_log)} steps):")
    for i, trace in enumerate(trace_log, 1):
        print(f"   {i}. {trace}")

# Step 6: 最終總結
print("\n" + "="*70)
print("🎉 Test Summary")
print("="*70)

success_count = 0
total_checks = 6

# 檢查 1: Workflow 執行
print("\n✓ Workflow execution: SUCCESS")
success_count += 1

# 檢查 2: 預測結果
if classification != "N/A":
    print("✓ Prediction result: SUCCESS")
    success_count += 1
else:
    print("✗ Prediction result: FAILED")

# 檢查 3: 信心分數
if confidence > 0:
    print("✓ Confidence score: SUCCESS")
    success_count += 1
else:
    print("✗ Confidence score: FAILED")

# 檢查 4: 特徵重要性
if feature_importances:
    print("✓ Feature importance: SUCCESS")
    success_count += 1
else:
    print("✗ Feature importance: FAILED")

# 檢查 5: 視覺化
if viz_paths and all(Path(p).exists() for p in viz_paths):
    print("✓ Visualizations: SUCCESS")
    success_count += 1
else:
    print("✗ Visualizations: FAILED")

# 檢查 6: 報告
if reports and "en" in reports and "zh" in reports:
    print("✓ Report generation: SUCCESS")
    success_count += 1
else:
    print("✗ Report generation: FAILED")

print(f"\n📊 Overall Score: {success_count}/{total_checks} checks passed")

if success_count == total_checks:
    print("\n🎉 ALL TESTS PASSED! System is working perfectly!")
    print("\n✨ Next steps:")
    print("   1. Review the generated visualizations")
    print("   2. Read the clinical reports")
    print("   3. Integrate into Streamlit UI (app.py)")
elif success_count >= 4:
    print("\n✅ MOSTLY WORKING! Some minor issues to fix.")
    print("\n🔧 Recommended actions:")
    print("   1. Check error logs above")
    print("   2. Fix any missing components")
    print("   3. Re-run the test")
else:
    print("\n⚠️  NEEDS ATTENTION! Several components failed.")
    print("\n🔧 Troubleshooting:")
    print("   1. Check error logs above")
    print("   2. Verify model files exist")
    print("   3. Check dependencies are installed")

print("\n" + "="*70)
