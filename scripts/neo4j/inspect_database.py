"""
Inspect Neo4j Database Contents

This script inspects what's actually in the Neo4j database to understand
why multi-hop queries are returning no results.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.knowledge.neo4j_dao import Neo4jDAO


def inspect_nodes():
    """Inspect what nodes exist in the database"""
    print("\n" + "="*80)
    print("INSPECTING NODES")
    print("="*80)
    
    dao = Neo4jDAO()
    
    # Count nodes by label
    query = """
    MATCH (n)
    RETURN DISTINCT labels(n) AS labels, count(n) AS count
    ORDER BY count DESC
    """
    
    results = dao._execute_read(query)
    
    print("\nNode counts by label:")
    for result in results:
        labels = result['labels']
        count = result['count']
        print(f"  {labels}: {count} nodes")
    
    dao.close()


def inspect_relationships():
    """Inspect what relationships exist in the database"""
    print("\n" + "="*80)
    print("INSPECTING RELATIONSHIPS")
    print("="*80)
    
    dao = Neo4jDAO()
    
    # Count relationships by type
    query = """
    MATCH ()-[r]->()
    RETURN type(r) AS relationship_type, count(r) AS count
    ORDER BY count DESC
    """
    
    results = dao._execute_read(query)
    
    print("\nRelationship counts by type:")
    if results:
        for result in results:
            rel_type = result['relationship_type']
            count = result['count']
            print(f"  {rel_type}: {count} relationships")
    else:
        print("  ⚠ NO RELATIONSHIPS FOUND IN DATABASE")
    
    dao.close()


def inspect_brain_regions():
    """Inspect BrainRegion nodes"""
    print("\n" + "="*80)
    print("INSPECTING BRAIN REGIONS")
    print("="*80)
    
    dao = Neo4jDAO()
    
    # Get sample brain regions
    query = """
    MATCH (r:BrainRegion)
    RETURN r.id AS id, r.name AS name, r.summary AS summary
    LIMIT 10
    """
    
    results = dao._execute_read(query)
    
    print(f"\nFound {len(results)} BrainRegion nodes (showing first 10):")
    for result in results:
        print(f"\n  ID: {result.get('id', 'N/A')}")
        print(f"  Name: {result.get('name', 'N/A')}")
        print(f"  Summary: {result.get('summary', 'N/A')[:60]}...")
    
    dao.close()


def inspect_sample_connections():
    """Inspect connections for a specific region"""
    print("\n" + "="*80)
    print("INSPECTING CONNECTIONS FOR HIPPOCAMPUS_L")
    print("="*80)
    
    dao = Neo4jDAO()
    
    # Check if Hippocampus_L exists
    query1 = """
    MATCH (r:BrainRegion)
    WHERE r.id = 'Hippocampus_L' OR r.name CONTAINS 'Hippocampus'
    RETURN r.id AS id, r.name AS name
    LIMIT 5
    """
    
    results1 = dao._execute_read(query1)
    
    print("\nHippocampus nodes found:")
    for result in results1:
        print(f"  ID: {result.get('id')}, Name: {result.get('name')}")
    
    if results1:
        # Check outgoing relationships
        query2 = """
        MATCH (r:BrainRegion)-[rel]->(target)
        WHERE r.id = 'Hippocampus_L' OR r.name CONTAINS 'Hippocampus'
        RETURN type(rel) AS rel_type, labels(target) AS target_labels, target.name AS target_name
        LIMIT 10
        """
        
        results2 = dao._execute_read(query2)
        
        print(f"\nOutgoing relationships: {len(results2)}")
        for result in results2:
            print(f"  -{result.get('rel_type')}-> {result.get('target_labels')}: {result.get('target_name')}")
        
        # Check incoming relationships
        query3 = """
        MATCH (source)-[rel]->(r:BrainRegion)
        WHERE r.id = 'Hippocampus_L' OR r.name CONTAINS 'Hippocampus'
        RETURN type(rel) AS rel_type, labels(source) AS source_labels, source.name AS source_name
        LIMIT 10
        """
        
        results3 = dao._execute_read(query3)
        
        print(f"\nIncoming relationships: {len(results3)}")
        for result in results3:
            print(f"  {result.get('source_labels')}: {result.get('source_name')} -{result.get('rel_type')}->")
    
    dao.close()


def inspect_diseases():
    """Inspect Disease nodes"""
    print("\n" + "="*80)
    print("INSPECTING DISEASES")
    print("="*80)
    
    dao = Neo4jDAO()
    
    # Get all diseases
    query = """
    MATCH (d:Disease)
    RETURN d.id AS id, d.name AS name
    """
    
    results = dao._execute_read(query)
    
    print(f"\nFound {len(results)} Disease nodes:")
    for result in results:
        print(f"  ID: {result.get('id')}, Name: {result.get('name')}")
    
    dao.close()


def main():
    """Run all inspections"""
    print("\n" + "="*80)
    print("NEO4J DATABASE INSPECTION")
    print("="*80)
    print("\nThis script will inspect the Neo4j database to understand")
    print("why multi-hop queries are returning no results.")
    
    try:
        inspect_nodes()
        inspect_relationships()
        inspect_brain_regions()
        inspect_diseases()
        inspect_sample_connections()
        
        print("\n" + "="*80)
        print("INSPECTION COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. If no relationships found: Run relationship ingestion script")
        print("  2. If relationships exist but wrong type: Update query templates")
        print("  3. If nodes missing: Run node ingestion script")
        
    except Exception as e:
        print(f"\n✗ INSPECTION FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
