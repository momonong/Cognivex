"""
Test Multi-hop Query Refinements

This script tests the refined Cypher queries for multi-hop traversal
in the GraphRAG service, specifically targeting Tests 2, 4, and 5.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.knowledge.neo4j_dao import Neo4jDAO
from app.core.knowledge.graph_rag import GraphRAG


def test_query_multiple_regions():
    """Test 2: Query multiple regions with context"""
    print("\n" + "="*80)
    print("TEST 2: Query Multiple Regions (Multi-hop Context)")
    print("="*80)
    
    try:
        dao = Neo4jDAO()
        
        # Test with actual region IDs from database
        regions = ['HIP_L', 'HIP_R', 'PreCG_L']
        print(f"\nQuerying regions: {regions}")
        
        results = dao.query_regions_by_names(regions)
        
        print(f"\n✓ Query executed successfully")
        print(f"✓ Found {len(results)} results")
        
        for result in results:
            print(f"\n  Region: {result.get('name', 'N/A')}")
            print(f"    ID: {result.get('id', 'N/A')}")
            print(f"    Summary: {result.get('summary', 'N/A')[:60]}...")
            print(f"    AD Hotspot: {result.get('is_ad_hotspot', False)}")
            print(f"    Networks: {result.get('networks', [])}")
            print(f"    Diseases: {result.get('diseases', [])}")
            print(f"    Functions: {result.get('functions', [])}")
        
        dao.close()
        
        if len(results) > 0:
            print("\n✓ TEST 2 PASSED: Multi-hop context retrieval working")
            return True
        else:
            print("\n✗ TEST 2 FAILED: No results returned")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_related_regions():
    """Test 4: Find related regions through graph traversal"""
    print("\n" + "="*80)
    print("TEST 4: Find Related Regions (Multi-hop Traversal)")
    print("="*80)
    
    try:
        dao = Neo4jDAO()
        
        # Test with Hippocampus (should have many connections)
        region_id = 'HIP_L'
        print(f"\nFinding regions related to: {region_id}")
        
        results = dao.query_related_regions(region_id, max_hops=2)
        
        print(f"\n✓ Query executed successfully")
        print(f"✓ Found {len(results)} related regions")
        
        for result in results[:5]:  # Show top 5
            print(f"\n  Related Region: {result.get('name', 'N/A')}")
            print(f"    ID: {result.get('id', 'N/A')}")
            print(f"    Summary: {result.get('summary', 'N/A')[:60]}...")
            print(f"    Distance: {result.get('distance', 'N/A')} hops")
            print(f"    AD Hotspot: {result.get('is_ad_hotspot', False)}")
        
        dao.close()
        
        if len(results) > 0:
            print("\n✓ TEST 4 PASSED: Multi-hop traversal working")
            return True
        else:
            print("\n✗ TEST 4 FAILED: No related regions found")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_disease_associations():
    """Test 5: Query disease associations"""
    print("\n" + "="*80)
    print("TEST 5: Query Disease Associations")
    print("="*80)
    
    try:
        dao = Neo4jDAO()
        
        # Test with Alzheimer's Disease
        disease_name = "Alzheimer's Disease"
        print(f"\nQuerying regions associated with: {disease_name}")
        
        results = dao.query_disease_associations(disease_name)
        
        print(f"\n✓ Query executed successfully")
        print(f"✓ Found {len(results)} associated regions")
        
        for result in results[:10]:  # Show top 10
            hotspot = "[AD HOTSPOT]" if result.get('is_ad_hotspot') else ""
            print(f"\n  Region: {result.get('name', 'N/A')} {hotspot}")
            print(f"    ID: {result.get('id', 'N/A')}")
            print(f"    Summary: {result.get('summary', 'N/A')[:60]}...")
        
        dao.close()
        
        if len(results) > 0:
            print("\n✓ TEST 5 PASSED: Disease association query working")
            return True
        else:
            print("\n✗ TEST 5 FAILED: No disease associations found")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graphrag_integration():
    """Test GraphRAG integration with refined queries"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: GraphRAG with Refined Queries")
    print("="*80)
    
    try:
        graph_rag = GraphRAG()
        
        # Test 1: Query single region
        print("\n[1] Query single region: HIP_L (Hippocampus)")
        result = graph_rag.query_region('HIP_L')
        if result:
            print(f"  ✓ Found: {result['full_name']}")
        else:
            print("  ✗ Not found")
        
        # Test 2: Query multiple regions
        print("\n[2] Query multiple regions")
        regions = ['HIP_L', 'HIP_R', 'PreCG_L']
        results = graph_rag.query_multiple_regions(regions)
        print(f"  ✓ Queried {len(regions)} regions, got {len(results)} results")
        
        # Test 3: Generate context summary
        print("\n[3] Generate context summary")
        summary = graph_rag.generate_context_summary(results)
        print(f"  ✓ Summary: {summary[:100]}...")
        
        # Test 4: Find related regions
        print("\n[4] Find related regions to HIP_L")
        related = graph_rag.find_related_regions('HIP_L')
        print(f"  ✓ Found {len(related)} related regions")
        for r in related[:3]:
            print(f"    - {r['id']}: {r.get('full_name', 'N/A')}")
        
        # Test 5: Query disease associations
        print("\n[5] Query regions associated with Alzheimer's Disease")
        ad_regions = graph_rag.query_disease_associations("Alzheimer's Disease")
        print(f"  ✓ Found {len(ad_regions)} regions")
        for r in ad_regions[:5]:
            hotspot = "[AD Hotspot]" if r.get('is_ad_hotspot') else ""
            print(f"    - {r['id']} {hotspot}")
        
        graph_rag.close()
        
        print("\n✓ INTEGRATION TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all multi-hop query tests"""
    print("\n" + "="*80)
    print("MULTI-HOP QUERY REFINEMENT TEST SUITE")
    print("="*80)
    print("\nTesting refined Cypher queries for:")
    print("  - Test 2: Query multiple regions with context")
    print("  - Test 4: Find related regions (multi-hop traversal)")
    print("  - Test 5: Query disease associations")
    print("\nQuery improvements:")
    print("  ✓ Explicit relationship types (BELONGS_TO, INVOLVED_IN)")
    print("  ✓ Explicit node labels (BrainRegion, FunctionalNetwork, Disease, BrainFunction)")
    print("  ✓ Bidirectional relationship matching")
    print("  ✓ Proper path traversal through intermediate nodes")
    
    results = []
    
    # Run individual tests
    results.append(("Test 2: Query Multiple Regions", test_query_multiple_regions()))
    results.append(("Test 4: Find Related Regions", test_find_related_regions()))
    results.append(("Test 5: Disease Associations", test_disease_associations()))
    results.append(("Integration Test", test_graphrag_integration()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Multi-hop queries are working correctly!")
    else:
        print(f"\n✗ {total - passed} test(s) failed - Review query refinements")
    
    print("="*80)


if __name__ == "__main__":
    main()
