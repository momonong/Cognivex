"""
測試 report_generator 的修復
"""

from app.agents.report_generator import format_regions_for_prompt

# Test case 1: regions with None values
test_regions_with_none = [
    {
        "region_name": "Hippocampus_L",
        "activation_score": 0.85,
        "associated_networks": None,  # This was causing the error
        "known_functions": None  # This was causing the error
    },
    {
        "region_name": "Amygdala_R",
        "activation_score": 0.72,
        "associated_networks": ["Default Mode Network"],
        "known_functions": ["Memory", "Emotion"]
    }
]

# Test case 2: regions with empty lists
test_regions_with_empty = [
    {
        "region_name": "Precuneus_L",
        "activation_score": 0.65,
        "associated_networks": [],
        "known_functions": []
    }
]

# Test case 3: regions with valid data
test_regions_valid = [
    {
        "region_name": "Thalamus_L",
        "activation_score": 0.58,
        "associated_networks": ["Salience Network", "Executive Control"],
        "known_functions": ["Sensory relay", "Motor control"]
    }
]

print("="*70)
print("Testing report_generator fix")
print("="*70)

print("\n[TEST 1] Regions with None values (was causing error)")
print("-" * 50)
try:
    result = format_regions_for_prompt(test_regions_with_none)
    print("✅ PASSED - No error with None values")
    print(f"\nOutput:\n{result}")
except Exception as e:
    print(f"❌ FAILED - Error: {e}")

print("\n[TEST 2] Regions with empty lists")
print("-" * 50)
try:
    result = format_regions_for_prompt(test_regions_with_empty)
    print("✅ PASSED - No error with empty lists")
    print(f"\nOutput:\n{result}")
except Exception as e:
    print(f"❌ FAILED - Error: {e}")

print("\n[TEST 3] Regions with valid data")
print("-" * 50)
try:
    result = format_regions_for_prompt(test_regions_valid)
    print("✅ PASSED - No error with valid data")
    print(f"\nOutput:\n{result}")
except Exception as e:
    print(f"❌ FAILED - Error: {e}")

print("\n" + "="*70)
print("All tests completed!")
print("="*70)
