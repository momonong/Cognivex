#!/usr/bin/env python3
"""
Test the complete LangGraph pipeline with fixed activation analysis
"""
import sys
sys.path.append('.')

from app.graph.workflow import app

def test_full_pipeline():
    try:
        print("Testing complete LangGraph pipeline with ShuffleNet...")
        
        initial_state = {
            "subject_id": "sub-01",
            "fmri_scan_path": "data/raw/CN/sub-01/dswausub-009_S_0751_task-rest_bold.nii.gz",
            "model_path": "model/shufflenet/fold_3_best_model.pth",
            "model_name": "shufflenet",
            "trace_log": [],
            "error_log": [],
        }
        
        print("Running complete LangGraph pipeline...")
        final_state = app.invoke(initial_state)
        
        if final_state:
            print("✅ Pipeline completed successfully!")
            
            # Check results
            classification = final_state.get("classification_result")
            print(f"  - Classification: {classification}")
            
            activated_regions = final_state.get("activated_regions", [])
            print(f"  - Activated regions: {len(activated_regions)}")
            
            if activated_regions:
                print("  - Top 5 regions:")
                for i, region in enumerate(activated_regions[:5]):
                    name = region.get('region_name', 'Unknown')
                    score = region.get('activation_score', 0)
                    hemisphere = region.get('hemisphere', 'Unknown')
                    networks = region.get('associated_networks', [])
                    functions = region.get('known_functions', [])
                    
                    print(f"    {i+1}. {name} (score: {score:.3f}, {hemisphere})")
                    if networks:
                        print(f"       Networks: {networks}")
                    if functions:
                        print(f"       Functions: {functions[:2]}...")  # Show first 2 functions
            
            clean_regions = final_state.get("clean_region_names", [])
            print(f"  - Clean region names: {len(clean_regions)}")
            
            image_explanation = final_state.get("image_explanation")
            if image_explanation:
                print(f"  - Image explanation available: Yes")
            
            reports = final_state.get("generated_reports", {})
            print(f"  - Generated reports: {list(reports.keys())}")
            
            errors = final_state.get("error_log", [])
            if errors:
                print(f"  - Errors: {len(errors)}")
                for error in errors:
                    print(f"    * {error}")
            
            return len(activated_regions) > 0 and len(clean_regions) > 0
        else:
            print("❌ Pipeline returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        print("\n🎉 Complete pipeline test PASSED!")
    else:
        print("\n💥 Complete pipeline test FAILED!")