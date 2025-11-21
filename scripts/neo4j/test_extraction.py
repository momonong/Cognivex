"""
Test script for relationship extraction logic

Tests the ID and relationship type extraction from Neo4j export format.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.neo4j.ingest_relationships_from_export import RelationshipIngester


def test_extraction():
    """Test extraction logic"""
    print("="*80)
    print("TESTING EXTRACTION LOGIC")
    print("="*80)
    
    # Create ingester (without connecting to Neo4j)
    class MockIngester:
        def extract_node_id(self, node_str):
            return RelationshipIngester.extract_node_id(None, node_str)
        
        def extract_relationship_type(self, rel_str):
            return RelationshipIngester.extract_relationship_type(None, rel_str)
    
    ingester = MockIngester()
    
    # Test cases
    test_cases = [
        {
            'name': 'BrainRegion with single quotes',
            'node': "(:BrainRegion {id: 'PreCG_L', name: 'Left Precentral Gyrus'})",
            'expected_id': 'PreCG_L'
        },
        {
            'name': 'Disease with double quotes',
            'node': '(:Disease {id: "AD", name: "Alzheimer\'s Disease"})',
            'expected_id': 'AD'
        },
        {
            'name': 'FunctionalNetwork',
            'node': "(:FunctionalNetwork {id: 'SMN', name: 'Sensorimotor Network'})",
            'expected_id': 'SMN'
        },
        {
            'name': 'Simple node',
            'node': "(:BrainRegion {id: 'Hippocampus_L'})",
            'expected_id': 'Hippocampus_L'
        }
    ]
    
    rel_test_cases = [
        {
            'name': 'BELONGS_TO',
            'rel': '[:BELONGS_TO]',
            'expected_type': 'BELONGS_TO'
        },
        {
            'name': 'INVOLVED_IN',
            'rel': '[:INVOLVED_IN]',
            'expected_type': 'INVOLVED_IN'
        },
        {
            'name': 'SUPPORTS',
            'rel': '[:SUPPORTS]',
            'expected_type': 'SUPPORTS'
        }
    ]
    
    # Test node ID extraction
    print("\n[TEST 1] Node ID Extraction")
    print("-"*80)
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        result = ingester.extract_node_id(test['node'])
        expected = test['expected_id']
        
        if result == expected:
            print(f"✓ {test['name']}: '{result}'")
            passed += 1
        else:
            print(f"✗ {test['name']}: Expected '{expected}', got '{result}'")
            failed += 1
    
    print(f"\nNode ID Tests: {passed} passed, {failed} failed")
    
    # Test relationship type extraction
    print("\n[TEST 2] Relationship Type Extraction")
    print("-"*80)
    
    rel_passed = 0
    rel_failed = 0
    
    for test in rel_test_cases:
        result = ingester.extract_relationship_type(test['rel'])
        expected = test['expected_type']
        
        if result == expected:
            print(f"✓ {test['name']}: '{result}'")
            rel_passed += 1
        else:
            print(f"✗ {test['name']}: Expected '{expected}', got '{result}'")
            rel_failed += 1
    
    print(f"\nRelationship Type Tests: {rel_passed} passed, {rel_failed} failed")
    
    # Overall summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total_passed = passed + rel_passed
    total_failed = failed + rel_failed
    print(f"Total: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = test_extraction()
    sys.exit(0 if success else 1)
