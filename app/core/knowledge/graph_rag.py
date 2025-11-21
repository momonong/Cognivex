"""
CDDA Framework - Knowledge Integration (Layer 4)

This module implements GraphRAG (Graph Retrieval-Augmented Generation)
for clinical knowledge retrieval from Neo4j knowledge graph.

Features:
- ROI-to-knowledge entity linking
- Multi-hop graph traversal
- Clinical context retrieval
- Disease association queries
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

try:
    from neo4j import GraphDatabase
    from app.core.knowledge.neo4j_dao import Neo4jDAO
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    Neo4jDAO = None
    print("[WARN] Neo4j driver not installed. GraphRAG will use fallback mode.")
    print("[WARN] Install with: pip install neo4j")


class GraphRAG:
    """
    Graph Retrieval-Augmented Generation
    
    Retrieves clinical knowledge from Neo4j knowledge graph
    to provide context for anomalous brain regions.
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        use_fallback: bool = False
    ):
        """
        Initialize GraphRAG
        
        Args:
            uri: Neo4j URI (default: from .env)
            user: Neo4j username (default: from .env)
            password: Neo4j password (default: from .env)
            use_fallback: Force fallback mode (mock data)
        """
        self.use_fallback = use_fallback or not NEO4J_AVAILABLE
        self.dao = None
        
        # Always initialize fallback data (for error recovery)
        self._init_fallback_data()
        
        if not self.use_fallback:
            # Try to initialize Neo4jDAO
            try:
                self.dao = Neo4jDAO(uri=uri, user=user, password=password)
                print(f"[OK] GraphRAG using Neo4jDAO (with fallback available)")
            except Exception as e:
                print(f"[WARN] Could not initialize Neo4jDAO: {e}")
                print("[WARN] Using fallback mode.")
                self.use_fallback = True
        
        if self.use_fallback:
            print("[INFO] GraphRAG running in fallback mode (mock data)")
    
    def _init_fallback_data(self):
        """Initialize fallback knowledge base (same as Phase 2 mock)"""
        self.fallback_kb = {
            'SN_pc': {
                'full_name': 'Substantia Nigra (pars compacta)',
                'function': 'Dopamine production, motor control',
                'clinical_significance': 'Atrophy associated with Parkinson\'s disease and mixed dementia',
                'related_conditions': ['Parkinson\'s Disease', 'Lewy Body Dementia', 'Mixed Dementia'],
                'is_ad_hotspot': False
            },
            'SN_pr': {
                'full_name': 'Substantia Nigra (pars reticulata)',
                'function': 'Motor control, basal ganglia output',
                'clinical_significance': 'Related to movement disorders',
                'related_conditions': ['Parkinson\'s Disease', 'Movement Disorders'],
                'is_ad_hotspot': False
            },
            'Hippocampus': {
                'full_name': 'Hippocampus',
                'function': 'Memory formation, spatial navigation',
                'clinical_significance': 'Early atrophy is hallmark of Alzheimer\'s disease',
                'related_conditions': ['Alzheimer\'s Disease', 'Mild Cognitive Impairment'],
                'is_ad_hotspot': True
            },
            'Hippocampus_L': {
                'full_name': 'Left Hippocampus',
                'function': 'Memory formation, verbal memory',
                'clinical_significance': 'Early atrophy is hallmark of Alzheimer\'s disease',
                'related_conditions': ['Alzheimer\'s Disease', 'Mild Cognitive Impairment'],
                'is_ad_hotspot': True
            },
            'Hippocampus_R': {
                'full_name': 'Right Hippocampus',
                'function': 'Memory formation, spatial memory',
                'clinical_significance': 'Early atrophy is hallmark of Alzheimer\'s disease',
                'related_conditions': ['Alzheimer\'s Disease', 'Mild Cognitive Impairment'],
                'is_ad_hotspot': True
            },
            'ACC': {
                'full_name': 'Anterior Cingulate Cortex',
                'function': 'Executive function, emotion regulation',
                'clinical_significance': 'Involved in cognitive control and decision making',
                'related_conditions': ['Frontotemporal Dementia', 'Depression'],
                'is_ad_hotspot': False
            },
            'Temporal': {
                'full_name': 'Temporal Lobe',
                'function': 'Memory, language, auditory processing',
                'clinical_significance': 'Atrophy common in Alzheimer\'s disease',
                'related_conditions': ['Alzheimer\'s Disease', 'Semantic Dementia'],
                'is_ad_hotspot': True
            },
            'Frontal': {
                'full_name': 'Frontal Lobe',
                'function': 'Executive function, planning, motor control',
                'clinical_significance': 'Atrophy in frontotemporal dementia',
                'related_conditions': ['Frontotemporal Dementia', 'Alzheimer\'s Disease'],
                'is_ad_hotspot': False
            }
        }
    
    def query_region(self, region_name: str) -> Optional[Dict]:
        """
        Query knowledge graph for a specific brain region
        
        Uses Neo4jDAO with parameterized queries for stability.
        
        Args:
            region_name: ROI name (e.g., 'Hippocampus_L')
        
        Returns:
            Dictionary with region information or None
        """
        if self.use_fallback or not self.dao:
            return self._query_region_fallback(region_name)
        
        try:
            # Use DAO for safe, parameterized query
            result = self.dao.query_region_by_id(region_name)
            
            if not result:
                # Try fallback if not found in Neo4j
                return self._query_region_fallback(region_name)
            
            return {
                'id': result.get('id'),
                'full_name': result.get('name', region_name),
                'summary': result.get('summary', 'No description available'),
                'is_ad_hotspot': result.get('is_ad_hotspot', False),
                'networks': result.get('networks', []),
                'functions': [],  # Not in current schema
                'related_conditions': result.get('diseases', [])
            }
                
        except Exception as e:
            print(f"[ERROR] Query failed for {region_name}: {e}")
            return self._query_region_fallback(region_name)
    
    def _query_region_fallback(self, region_name: str) -> Optional[Dict]:
        """Fallback query using mock data"""
        # Try exact match first
        if region_name in self.fallback_kb:
            data = self.fallback_kb[region_name].copy()
            data['id'] = region_name
            return data
        
        # Try partial match
        for key, value in self.fallback_kb.items():
            if key in region_name or region_name in key:
                data = value.copy()
                data['id'] = region_name
                data['full_name'] = region_name
                return data
        
        # No match found
        return {
            'id': region_name,
            'full_name': region_name,
            'function': 'Unknown',
            'clinical_significance': 'Requires further investigation',
            'related_conditions': [],
            'is_ad_hotspot': False
        }
    
    def query_multiple_regions(
        self,
        region_names: List[str],
        max_results: int = 10
    ) -> List[Dict]:
        """
        Query multiple brain regions
        
        Uses Neo4jDAO batch query for efficiency.
        
        Args:
            region_names: List of ROI names
            max_results: Maximum number of results to return
        
        Returns:
            List of region information dictionaries
        """
        if self.use_fallback or not self.dao:
            # Fallback: query one by one
            results = []
            for region_name in region_names[:max_results]:
                region_info = self.query_region(region_name)
                if region_info:
                    results.append(region_info)
            return results
        
        try:
            # Use DAO batch query (more efficient)
            dao_results = self.dao.query_regions_by_names(region_names[:max_results])
            
            # Group results by region
            region_map = {}
            for record in dao_results:
                region_id = record.get('id')
                if region_id not in region_map:
                    region_map[region_id] = {
                        'id': region_id,
                        'full_name': record.get('name', region_id),
                        'summary': record.get('summary', 'No description available'),
                        'is_ad_hotspot': record.get('is_ad_hotspot', False),
                        'related_conditions': []
                    }
            
            return list(region_map.values())
            
        except Exception as e:
            print(f"[ERROR] Batch query failed: {e}")
            # Fallback to one-by-one query
            results = []
            for region_name in region_names[:max_results]:
                region_info = self.query_region(region_name)
                if region_info:
                    results.append(region_info)
            return results
    
    def find_related_regions(
        self,
        region_name: str,
        relationship_type: Optional[str] = None,
        max_hops: int = 2
    ) -> List[Dict]:
        """
        Find regions related to a given region through graph traversal
        
        Uses Neo4jDAO with parameterized queries.
        
        Args:
            region_name: Starting ROI name
            relationship_type: Specific relationship to follow (optional)
            max_hops: Maximum number of hops in graph traversal
        
        Returns:
            List of related region information
        """
        if self.use_fallback or not self.dao:
            # Fallback: return regions with same disease associations
            source = self.query_region(region_name)
            if not source or not source.get('related_conditions'):
                return []
            
            related = []
            for key, value in self.fallback_kb.items():
                if key != region_name:
                    # Check for shared conditions
                    shared = set(source['related_conditions']) & set(value.get('related_conditions', []))
                    if shared:
                        info = value.copy()
                        info['id'] = key
                        info['shared_conditions'] = list(shared)
                        related.append(info)
            
            return related[:5]
        
        try:
            # Use DAO for safe, parameterized query
            dao_results = self.dao.query_related_regions(region_name, max_hops=max_hops)
            
            related_regions = []
            for record in dao_results:
                related_regions.append({
                    'id': record.get('id'),
                    'full_name': record.get('name'),
                    'summary': record.get('summary'),
                    'is_ad_hotspot': record.get('is_ad_hotspot', False),
                    'distance': record.get('distance', 0)
                })
            
            return related_regions
                
        except Exception as e:
            print(f"[ERROR] Related regions query failed: {e}")
            return []
    
    def query_disease_associations(
        self,
        disease_name: str
    ) -> List[Dict]:
        """
        Query all brain regions associated with a disease
        
        Uses Neo4jDAO with parameterized queries.
        
        Args:
            disease_name: Disease name (e.g., 'Alzheimer\'s Disease')
        
        Returns:
            List of associated brain regions
        """
        if self.use_fallback or not self.dao:
            # Fallback: filter by disease in related_conditions
            results = []
            for key, value in self.fallback_kb.items():
                if disease_name in value.get('related_conditions', []):
                    info = value.copy()
                    info['id'] = key
                    results.append(info)
            return results
        
        try:
            # Use DAO for safe, parameterized query
            dao_results = self.dao.query_disease_associations(disease_name)
            
            regions = []
            for record in dao_results:
                regions.append({
                    'id': record.get('id'),
                    'full_name': record.get('name'),
                    'summary': record.get('summary'),
                    'is_ad_hotspot': record.get('is_ad_hotspot', False)
                })
            
            return regions
                
        except Exception as e:
            print(f"[ERROR] Disease association query failed: {e}")
            return []
    
    def generate_context_summary(
        self,
        regions: List[Dict]
    ) -> str:
        """
        Generate natural language summary from region contexts
        
        Args:
            regions: List of region information dictionaries
        
        Returns:
            Natural language summary string
        """
        if not regions:
            return "No clinical context available."
        
        summaries = []
        
        for region in regions:
            region_id = region.get('id', 'Unknown')
            full_name = region.get('full_name', region_id)
            
            # Build summary based on available information
            if 'clinical_significance' in region:
                sig = region['clinical_significance']
            elif 'summary' in region:
                sig = region['summary']
            else:
                sig = "No description available"
            
            # Add disease associations
            conditions = region.get('related_conditions', [])
            if conditions:
                conditions_str = ', '.join(conditions[:2])
                summary = f"{full_name}: {sig}. Related to {conditions_str}."
            else:
                summary = f"{full_name}: {sig}."
            
            # Add AD hotspot indicator
            if region.get('is_ad_hotspot'):
                summary += " [AD Hotspot]"
            
            summaries.append(summary)
        
        return ' '.join(summaries)
    
    def close(self):
        """Close Neo4j connection"""
        if not self.use_fallback and self.dao:
            self.dao.close()
            print("[OK] GraphRAG connection closed")


def demo_graphrag():
    """Demo: GraphRAG functionality"""
    print("\n" + "="*80)
    print("DEMO: GraphRAG - Knowledge Retrieval")
    print("="*80)
    
    # Initialize GraphRAG
    graph_rag = GraphRAG()
    
    # Test 1: Query single region
    print("\n[Test 1] Query single region: Hippocampus_L")
    result = graph_rag.query_region('Hippocampus_L')
    if result:
        print(f"  ID: {result['id']}")
        print(f"  Name: {result['full_name']}")
        print(f"  Function: {result.get('function', result.get('summary', 'N/A'))}")
        print(f"  AD Hotspot: {result.get('is_ad_hotspot', False)}")
        print(f"  Related Conditions: {result.get('related_conditions', [])}")
    
    # Test 2: Query multiple regions
    print("\n[Test 2] Query multiple regions")
    regions = ['Hippocampus_L', 'SN_pc', 'ACC']
    results = graph_rag.query_multiple_regions(regions)
    print(f"  Queried {len(regions)} regions, got {len(results)} results")
    
    # Test 3: Generate context summary
    print("\n[Test 3] Generate context summary")
    summary = graph_rag.generate_context_summary(results)
    print(f"  Summary: {summary}")
    
    # Test 4: Find related regions
    print("\n[Test 4] Find related regions to Hippocampus_L")
    related = graph_rag.find_related_regions('Hippocampus_L')
    print(f"  Found {len(related)} related regions:")
    for r in related[:3]:
        print(f"    - {r['id']}: {r.get('full_name', 'N/A')}")
    
    # Test 5: Query disease associations
    print("\n[Test 5] Query regions associated with Alzheimer's Disease")
    ad_regions = graph_rag.query_disease_associations("Alzheimer's Disease")
    print(f"  Found {len(ad_regions)} regions:")
    for r in ad_regions[:5]:
        hotspot = "[AD Hotspot]" if r.get('is_ad_hotspot') else ""
        print(f"    - {r['id']} {hotspot}")
    
    # Close connection
    graph_rag.close()
    
    print("\n" + "="*80)
    print("[SUCCESS] GraphRAG demo complete")
    print("="*80)


if __name__ == "__main__":
    demo_graphrag()
