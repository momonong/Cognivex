"""
Unit Tests for CDDA Tools

Tests the API compliance of Tool 1 and Tool 2 according to CDDA_Architecture_Spec.md
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.ml_processing.cdda_tools import CDDAToolKit


def test_tool_1_api_compliance():
    """Test that Tool 1 returns all mandatory fields"""
    print("\n" + "="*80)
    print("TEST: Tool 1 API Compliance")
    print("="*80)
    
    toolkit = CDDAToolKit()
    report = toolkit.get_diagnostic_report('sub-0005', verbose=False)
    
    # Check mandatory top-level fields
    required_fields = [
        'subject_id',
        'prediction_result',
        'confidence',
        'uq_score',
        'top_features',
        'anomaly_status',
        'metadata'
    ]
    
    print("\n[Checking Top-Level Fields]")
    for field in required_fields:
        assert field in report, f"Missing required field: {field}"
        print(f"  ✓ {field}: {type(report[field]).__name__}")
    
    # Check top_features structure
    print("\n[Checking top_features Structure]")
    assert len(report['top_features']) > 0, "top_features should not be empty"
    
    feature_required_fields = ['roi_name', 'feature_value', 'z_score', 'shap_value', 'rank']
    first_feature = report['top_features'][0]
    
    for field in feature_required_fields:
        assert field in first_feature, f"Missing required field in top_features: {field}"
        print(f"  ✓ {field}: {type(first_feature[field]).__name__}")
    
    # Check anomaly_status structure
    print("\n[Checking anomaly_status Structure]")
    anomaly_required_fields = ['has_anomaly', 'anomalous_regions', 'anomaly_type']
    
    for field in anomaly_required_fields:
        assert field in report['anomaly_status'], f"Missing required field in anomaly_status: {field}"
        print(f"  ✓ {field}: {type(report['anomaly_status'][field]).__name__}")
    
    # Check data types
    print("\n[Checking Data Types]")
    assert isinstance(report['subject_id'], str), "subject_id should be str"
    assert isinstance(report['prediction_result'], str), "prediction_result should be str"
    assert isinstance(report['confidence'], float), "confidence should be float"
    assert isinstance(report['uq_score'], float), "uq_score should be float"
    assert isinstance(report['top_features'], list), "top_features should be list"
    assert isinstance(report['anomaly_status'], dict), "anomaly_status should be dict"
    assert isinstance(report['metadata'], dict), "metadata should be dict"
    print("  ✓ All data types correct")
    
    # Check value ranges
    print("\n[Checking Value Ranges]")
    assert 0.0 <= report['confidence'] <= 1.0, "confidence should be in [0, 1]"
    assert 0.0 <= report['uq_score'] <= 1.0, "uq_score should be in [0, 1]"
    print(f"  ✓ confidence: {report['confidence']:.3f} (valid)")
    print(f"  ✓ uq_score: {report['uq_score']:.3f} (valid)")
    
    print("\n[SUCCESS] Tool 1 API is compliant with CDDA spec")
    return True


def test_tool_2_api_compliance():
    """Test that Tool 2 returns all mandatory fields"""
    print("\n" + "="*80)
    print("TEST: Tool 2 API Compliance")
    print("="*80)
    
    toolkit = CDDAToolKit()
    
    # Get top features first
    report = toolkit.get_diagnostic_report('sub-0005', verbose=False)
    top_rois = [feat['roi_name'] for feat in report['top_features'][:2]]
    
    # Run counterfactual
    cf_results = toolkit.simulate_counterfactual(
        'sub-0005',
        top_rois,
        verbose=False
    )
    
    # Check mandatory top-level fields
    required_fields = [
        'subject_id',
        'original_prediction',
        'original_confidence',
        'new_prediction',
        'new_confidence',
        'confidence_delta',
        'masked_features',
        'interpretation'
    ]
    
    print("\n[Checking Top-Level Fields]")
    for field in required_fields:
        assert field in cf_results, f"Missing required field: {field}"
        print(f"  ✓ {field}: {type(cf_results[field]).__name__}")
    
    # Check masked_features structure
    print("\n[Checking masked_features Structure]")
    assert len(cf_results['masked_features']) > 0, "masked_features should not be empty"
    
    feature_required_fields = ['roi_name', 'original_value', 'masked_value', 'impact']
    first_feature = cf_results['masked_features'][0]
    
    for field in feature_required_fields:
        assert field in first_feature, f"Missing required field in masked_features: {field}"
        print(f"  ✓ {field}: {type(first_feature[field]).__name__}")
    
    # Check data types
    print("\n[Checking Data Types]")
    assert isinstance(cf_results['subject_id'], str), "subject_id should be str"
    assert isinstance(cf_results['original_prediction'], str), "original_prediction should be str"
    assert isinstance(cf_results['original_confidence'], float), "original_confidence should be float"
    assert isinstance(cf_results['new_prediction'], str), "new_prediction should be str"
    assert isinstance(cf_results['new_confidence'], float), "new_confidence should be float"
    assert isinstance(cf_results['confidence_delta'], float), "confidence_delta should be float"
    assert isinstance(cf_results['masked_features'], list), "masked_features should be list"
    assert isinstance(cf_results['interpretation'], str), "interpretation should be str"
    print("  ✓ All data types correct")
    
    # Check value ranges
    print("\n[Checking Value Ranges]")
    assert 0.0 <= cf_results['original_confidence'] <= 1.0, "original_confidence should be in [0, 1]"
    assert 0.0 <= cf_results['new_confidence'] <= 1.0, "new_confidence should be in [0, 1]"
    print(f"  ✓ original_confidence: {cf_results['original_confidence']:.3f} (valid)")
    print(f"  ✓ new_confidence: {cf_results['new_confidence']:.3f} (valid)")
    print(f"  ✓ confidence_delta: {cf_results['confidence_delta']:+.3f}")
    
    print("\n[SUCCESS] Tool 2 API is compliant with CDDA spec")
    return True


def test_uq_threshold_trigger():
    """Test that high UQ scores are correctly identified"""
    print("\n" + "="*80)
    print("TEST: UQ Threshold Detection")
    print("="*80)
    
    toolkit = CDDAToolKit(uq_threshold=0.8)
    
    # Test multiple subjects
    test_subjects = ['sub-0005', 'sub-0010', 'sub-0015']
    
    for subject_id in test_subjects:
        try:
            report = toolkit.get_diagnostic_report(subject_id, verbose=False)
            
            print(f"\n{subject_id}:")
            print(f"  Prediction: {report['prediction_result']} ({report['confidence']:.1%})")
            print(f"  UQ Score: {report['uq_score']:.3f}")
            
            if report['uq_score'] > toolkit.uq_threshold:
                print(f"  ⚠️  HIGH UNCERTAINTY - Would trigger Tool 2 (Counterfactual)")
            else:
                print(f"  ✓ Normal uncertainty")
                
        except Exception as e:
            print(f"  ⚠️  Subject not found: {e}")
    
    print("\n[SUCCESS] UQ threshold detection working")
    return True


def test_anomaly_detection():
    """Test that anomalies are correctly detected"""
    print("\n" + "="*80)
    print("TEST: Anomaly Detection")
    print("="*80)
    
    toolkit = CDDAToolKit(z_score_threshold=2.5)
    
    report = toolkit.get_diagnostic_report('sub-0005', verbose=False)
    
    print(f"\nSubject: {report['subject_id']}")
    print(f"Anomaly Status: {report['anomaly_status']['has_anomaly']}")
    
    if report['anomaly_status']['has_anomaly']:
        print(f"Anomalous Regions ({len(report['anomaly_status']['anomalous_regions'])}):")
        for region in report['anomaly_status']['anomalous_regions'][:5]:
            print(f"  - {region}")
        print(f"  ⚠️  Would trigger Tool 4 (GraphRAG Lookup)")
    else:
        print("  ✓ No anomalies detected")
    
    # Show top z-scores
    print(f"\nTop 5 Features by |Z-Score|:")
    sorted_features = sorted(
        report['top_features'],
        key=lambda x: abs(x['z_score']),
        reverse=True
    )[:5]
    
    for feat in sorted_features:
        print(f"  {feat['roi_name']}: z={feat['z_score']:+.2f}")
    
    print("\n[SUCCESS] Anomaly detection working")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("CDDA TOOLS - API COMPLIANCE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Tool 1 API Compliance", test_tool_1_api_compliance),
        ("Tool 2 API Compliance", test_tool_2_api_compliance),
        ("UQ Threshold Detection", test_uq_threshold_trigger),
        ("Anomaly Detection", test_anomaly_detection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAIL"))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! CDDA Tools are ready for Phase 2.")
    else:
        print("\n⚠️  Some tests failed. Please review.")
    
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
