"""
Test End-to-End CNN-RF Inference

This script tests the CNN-RF end-to-end inference agent
that processes raw MRI images directly.

Usage:
    python app/test_end_to_end_inference.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cnn_rf_inference import run_cnn_rf_inference, run_cnn_rf_inference_with_visualization


def test_basic_inference():
    """Test basic end-to-end CNN-RF inference"""
    print("\n" + "="*80)
    print("TEST 1: End-to-End CNN-RF Inference")
    print("="*80)
    
    # Create test state
    state = {
        'subject_id': 'sub-0005',
        'model_name': 'NC_vs_AD',
        'data_root': 'data/MRI_processed',
        'trace_log': [],
        'error_log': []
    }
    
    # Run inference
    result = run_cnn_rf_inference(state)
    
    # Check results
    print("\n[RESULTS]")
    print(f"Classification: {result.get('classification_result')}")
    print(f"Confidence: {result.get('prediction_confidence', 0):.1%}")
    print(f"Ground Truth: {result.get('true_label')}")
    print(f"Correct: {result.get('correct_prediction')}")
    print(f"Probabilities: {result.get('prediction_probabilities')}")
    print(f"Subject Directory: {result.get('subject_directory')}")
    print(f"Model: {result.get('model_name')}")
    
    if result.get('error_log'):
        print(f"\n[ERRORS]")
        for error in result['error_log']:
            print(f"  - {error}")
    
    return result


def test_inference_with_visualization():
    """Test end-to-end CNN-RF inference with brain visualization"""
    print("\n" + "="*80)
    print("TEST 2: End-to-End CNN-RF Inference with Visualization")
    print("="*80)
    
    # Create test state
    state = {
        'subject_id': 'sub-0010',
        'model_name': 'NC_vs_AD',
        'data_root': 'data/MRI_processed',
        'trace_log': [],
        'error_log': []
    }
    
    # Run inference with visualization
    result = run_cnn_rf_inference_with_visualization(state)
    
    # Check results
    print("\n[RESULTS]")
    print(f"Classification: {result.get('classification_result')}")
    print(f"Confidence: {result.get('prediction_confidence', 0):.1%}")
    print(f"Ground Truth: {result.get('true_label')}")
    print(f"Correct: {result.get('correct_prediction')}")
    print(f"Brain Map: {result.get('brain_map_path', 'Not generated')}")
    
    if result.get('error_log'):
        print(f"\n[ERRORS]")
        for error in result['error_log']:
            print(f"  - {error}")
    
    return result


def test_multiple_subjects():
    """Test end-to-end inference on multiple subjects"""
    print("\n" + "="*80)
    print("TEST 3: Multiple Subject End-to-End Inference")
    print("="*80)
    
    subjects = ['sub-0005', 'sub-0010', 'sub-0015']
    results = []
    
    for subject_id in subjects:
        print(f"\n[Testing {subject_id}]")
        
        state = {
            'subject_id': subject_id,
            'model_name': 'NC_vs_AD',
            'data_root': 'data/MRI_processed',
            'trace_log': [],
            'error_log': []
        }
        
        result = run_cnn_rf_inference(state)
        results.append(result)
        
        status = "✓" if result.get('correct_prediction') else "✗"
        print(f"  {status} Result: {result.get('classification_result')} "
              f"(confidence: {result.get('prediction_confidence', 0):.1%}, "
              f"ground truth: {result.get('true_label')})")
    
    # Summary
    print("\n[SUMMARY]")
    print(f"Total subjects tested: {len(subjects)}")
    successful = sum(1 for r in results if 'ERROR' not in r.get('classification_result', ''))
    correct = sum(1 for r in results if r.get('correct_prediction', False))
    print(f"Successful predictions: {successful}/{len(subjects)}")
    print(f"Correct predictions: {correct}/{len(subjects)}")
    if successful > 0:
        print(f"Accuracy: {correct/successful:.1%}")
    
    return results


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CNN-RF End-to-End Inference Test Suite")
    print("="*80)
    
    # Test 1: Basic inference
    test_basic_inference()
    
    # Test 2: Inference with visualization
    test_inference_with_visualization()
    
    # Test 3: Multiple subjects
    test_multiple_subjects()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()
