#!/usr/bin/env python3
"""
Model System Test Suite

This script runs all tests for the generic model inference system.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_test_script(script_name, description):
    """Run a test script and capture results"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}")
    
    script_path = Path(__file__).parent / script_name
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        if result.returncode == 0:
            print("✅ PASSED")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr:
                print("\nError output:")
                print(result.stderr)
            if result.stdout:
                print("\nStandard output:")
                print(result.stdout)
                
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ FAILED to run script: {e}")
        return False

def main():
    """Run all model system tests"""
    print("Generic Model System Test Suite")
    print("=" * 50)
    print("Testing the generic model inference system components")
    
    tests = [
        ("analyze_layers.py", "Layer Analysis & Recommendations"),
        ("test_layer_selection.py", "Detailed Layer Selection Testing"),
        ("test_improved_selection.py", "Improved Strategy Testing"),
    ]
    
    results = {}
    
    for script, description in tests:
        results[script] = run_test_script(script, description)
    
    # Summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    print("-" * 40)
    
    for script, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{script:30} | {status}")
    
    if passed == total:
        print(f"\n🎉 All tests passed! The generic model system is working correctly.")
        print("\nNext steps:")
        print("• Run inference with real fMRI data")
        print("• Generate and review activation maps") 
        print("• Fine-tune layer selection if needed")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)