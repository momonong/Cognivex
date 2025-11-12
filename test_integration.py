"""
Quick integration test for ML model
"""

print("="*60)
print("Testing ML Model Integration")
print("="*60)

# Test 1: Model Loader
print("\n[Test 1] Testing Model Loader...")
try:
    from app.core.ml_processing import MLModelLoader
    loader = MLModelLoader()
    components = loader.get_all_components()
    print(f"✓ Model loaded successfully!")
    print(f"  - Features: {len(components['feature_names'])}")
    print(f"  - Model type: {type(components['model']).__name__}")
except Exception as e:
    print(f"✗ Model loader failed: {e}")

# Test 2: Feature Extractor
print("\n[Test 2] Testing Feature Extractor...")
try:
    from app.core.ml_processing import ROIFeatureExtractor
    extractor = ROIFeatureExtractor()
    atlas_img, labels = extractor.load_atlas()
    print(f"✓ Atlas loaded successfully!")
    print(f"  - Total regions: {len(labels)}")
except Exception as e:
    print(f"✗ Feature extractor failed: {e}")

# Test 3: Workflow
print("\n[Test 3] Testing Workflow...")
try:
    from app.graph.workflow import app, route_by_analysis_mode
    from app.graph.state import AgentState
    
    # Test routing
    test_state = {"analysis_mode": "structural"}
    next_node = route_by_analysis_mode(test_state)
    print(f"✓ Workflow routing works!")
    print(f"  - Structural mode routes to: {next_node}")
    
    test_state = {"analysis_mode": "functional"}
    next_node = route_by_analysis_mode(test_state)
    print(f"  - Functional mode routes to: {next_node}")
except Exception as e:
    print(f"✗ Workflow test failed: {e}")

# Test 4: UI Components
print("\n[Test 4] Testing UI Components...")
try:
    from app.ui import (
        render_analysis_mode_selector,
        render_ml_model_info,
        render_structural_results
    )
    print(f"✓ UI components imported successfully!")
except Exception as e:
    print(f"✗ UI components failed: {e}")

print("\n" + "="*60)
print("Integration Test Complete!")
print("="*60)
