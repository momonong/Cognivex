"""
Test CNN-RF Integration

This script tests the CNN-RF model integration in the app workflow.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.graph.workflow import app
from app.graph.state import AgentState


def test_cnn_rf_inference():
    """Test CNN-RF inference with a sample subject"""
    
    print("="*80)
    print("Testing CNN-RF Integration")
    print("="*80)
    
    # Test subject (from roi_features.csv)
    subject_id = "sub-0005"  # AD patient
    
    # Initial state for CNN-RF inference
    initial_state = {
        "subject_id": subject_id,
        "analysis_mode": "structural",  # Use structural MRI branch
        "model_type": "cnn_rf",         # Use CNN-RF model
        "model_name": "NC_vs_AD",       # Binary classification
        "trace_log": [],
        "error_log": [],
    }
    
    print(f"\n[START] Starting CNN-RF pipeline for subject: {subject_id}")
    print(f"   Analysis mode: {initial_state['analysis_mode']}")
    print(f"   Model type: {initial_state['model_type']}")
    print(f"   Model name: {initial_state['model_name']}")
    print("="*80)
    
    try:
        # Execute the workflow
        final_state = app.invoke(initial_state)
        
        print("\n" + "="*80)
        print("[SUCCESS] Pipeline completed successfully!")
        print("="*80)
        
        # Display key results
        print("\n[RESULTS]")
        print(f"   Classification: {final_state.get('classification_result')}")
        print(f"   Confidence: {final_state.get('prediction_confidence', 0):.1%}")
        
        probabilities = final_state.get('prediction_probabilities', {})
        if probabilities:
            print(f"   Probabilities:")
            for cls, prob in probabilities.items():
                print(f"      {cls}: {prob:.1%}")
        
        important_rois = final_state.get('important_rois', [])
        if important_rois:
            print(f"\n[BRAIN REGIONS] Top 5 Important:")
            for i, roi in enumerate(important_rois[:5], 1):
                print(f"      {i}. {roi}")
        
        brain_map = final_state.get('brain_map_path')
        if brain_map:
            print(f"\n[VISUALIZATION] Brain map saved: {brain_map}")
        
        # Display trace log
        trace_log = final_state.get('trace_log', [])
        if trace_log:
            print(f"\n[TRACE LOG]")
            for i, trace in enumerate(trace_log, 1):
                print(f"   {i}. {trace}")
        
        # Display errors if any
        error_log = final_state.get('error_log', [])
        if error_log:
            print(f"\n[ERRORS]")
            for i, error in enumerate(error_log, 1):
                print(f"   {i}. {error}")
        
        return final_state
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        
        import traceback
        traceback.print_exc()
        
        return None


def test_multiple_subjects():
    """Test CNN-RF inference with multiple subjects"""
    
    print("\n" + "="*80)
    print("Testing Multiple Subjects")
    print("="*80)
    
    # Test subjects from different groups
    test_subjects = [
        ("sub-0005", "AD"),   # Alzheimer's Disease
        ("sub-0002", "NC"),   # Normal Control
    ]
    
    results = []
    
    for subject_id, expected_class in test_subjects:
        print(f"\n{'='*80}")
        print(f"Testing subject: {subject_id} (Expected: {expected_class})")
        print("="*80)
        
        initial_state = {
            "subject_id": subject_id,
            "analysis_mode": "structural",
            "model_type": "cnn_rf",
            "model_name": "NC_vs_AD",
            "trace_log": [],
            "error_log": [],
        }
        
        try:
            final_state = app.invoke(initial_state)
            
            prediction = final_state.get('classification_result')
            confidence = final_state.get('prediction_confidence', 0)
            
            print(f"\n   Prediction: {prediction}")
            print(f"   Expected: {expected_class}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Match: {'✓' if prediction == expected_class else '✗'}")
            
            results.append({
                'subject_id': subject_id,
                'expected': expected_class,
                'predicted': prediction,
                'confidence': confidence,
                'match': prediction == expected_class
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'subject_id': subject_id,
                'expected': expected_class,
                'predicted': 'ERROR',
                'confidence': 0,
                'match': False
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print("="*80)
    
    correct = sum(1 for r in results if r['match'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"\nDetailed Results:")
    for r in results:
        status = "✓" if r['match'] else "✗"
        print(f"   {status} {r['subject_id']}: {r['predicted']} "
              f"(expected: {r['expected']}, confidence: {r['confidence']:.1%})")
    
    return results


def compare_models():
    """Compare legacy model vs CNN-RF model"""
    
    print("\n" + "="*80)
    print("Comparing Legacy Model vs CNN-RF Model")
    print("="*80)
    
    subject_id = "sub-0005"
    
    # Test legacy model
    print(f"\n[1/2] Testing Legacy Model...")
    legacy_state = {
        "subject_id": subject_id,
        "analysis_mode": "structural",
        "model_type": "legacy",
        "fmri_scan_path": "data/sMRI/AD/sub-0005/sub_0005_T1.nii.gz",
        "trace_log": [],
        "error_log": [],
    }
    
    try:
        legacy_result = app.invoke(legacy_state)
        legacy_prediction = legacy_result.get('classification_result')
        legacy_confidence = legacy_result.get('prediction_confidence', 0)
        print(f"   Legacy: {legacy_prediction} ({legacy_confidence:.1%})")
    except Exception as e:
        print(f"   Legacy failed: {e}")
        legacy_prediction = "ERROR"
        legacy_confidence = 0
    
    # Test CNN-RF model
    print(f"\n[2/2] Testing CNN-RF Model...")
    cnn_rf_state = {
        "subject_id": subject_id,
        "analysis_mode": "structural",
        "model_type": "cnn_rf",
        "model_name": "NC_vs_AD",
        "trace_log": [],
        "error_log": [],
    }
    
    try:
        cnn_rf_result = app.invoke(cnn_rf_state)
        cnn_rf_prediction = cnn_rf_result.get('classification_result')
        cnn_rf_confidence = cnn_rf_result.get('prediction_confidence', 0)
        print(f"   CNN-RF: {cnn_rf_prediction} ({cnn_rf_confidence:.1%})")
    except Exception as e:
        print(f"   CNN-RF failed: {e}")
        cnn_rf_prediction = "ERROR"
        cnn_rf_confidence = 0
    
    # Comparison
    print(f"\n{'='*80}")
    print("Comparison")
    print("="*80)
    print(f"   Legacy:  {legacy_prediction} ({legacy_confidence:.1%})")
    print(f"   CNN-RF:  {cnn_rf_prediction} ({cnn_rf_confidence:.1%})")
    print(f"   Match:   {'✓' if legacy_prediction == cnn_rf_prediction else '✗'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CNN-RF Integration")
    parser.add_argument(
        '--mode',
        choices=['single', 'multiple', 'compare'],
        default='single',
        help='Test mode: single subject, multiple subjects, or compare models'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        test_cnn_rf_inference()
    elif args.mode == 'multiple':
        test_multiple_subjects()
    elif args.mode == 'compare':
        compare_models()
