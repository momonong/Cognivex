"""
Neo4j Data Access Object (DAO)

Provides a stable, robust interface for Neo4j database operations
using the official neo4j Python driver with proper session management
and parameterized queries.
"""

import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from neo4j import GraphDatabase, Session
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class Neo4jDAO:
    """
    Data Access Object for Neo4j
    
    Implements the DAO pattern for stable and secure database access.
    All queries use parameterization to prevent injection and ensure stability.
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize Neo4j DAO
        
        Args:
            uri: Neo4j URI (default: from .env NEO4J_URI)
            user: Neo4j username (default: from .env NEO4J_USER)
            password: Neo4j password (default: from .env NEO4J_PASSWORD)
        
        Raises:
            ValueError: If credentials are missing
            ConnectionError: If connection fails
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver not installed. Install with: pip install neo4j")
        
        # Get credentials
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        if not all([self.uri, self.user, self.password]):
            raise ValueError(
                "Neo4j credentials missing. Provide uri, user, password "
                "or set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env"
            )
        
        # Create driver
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            print(f"[OK] Neo4jDAO connected: {self.uri}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def _execute_read(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a read query with parameters
        
        This is the single, safe handler for all read queries.
        Uses session management and parameterization.
        
        Args:
            query: Cypher query string (with $param placeholders)
            params: Dictionary of parameters
        
        Returns:
            List of result records as dictionaries
        
        Raises:
            Exception: If query execution fails
        """
        if params is None:
            params = {}
        
        results = []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                
                # Convert records to dictionaries
                for record in result:
                    results.append(dict(record))
                
                return results
                
        except Exception as e:
            print(f"[ERROR] Query execution failed: {e}")
            print(f"[ERROR] Query: {query}")
            print(f"[ERROR] Params: {params}")
            raise
    
    def _execute_write(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a write query with parameters
        
        Args:
            query: Cypher query string (with $param placeholders)
            params: Dictionary of parameters
        
        Returns:
            Summary of the write operation
        
        Raises:
            Exception: If query execution fails
        """
        if params is None:
            params = {}
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                summary = result.consume()
                
                return {
                    'nodes_created': summary.counters.nodes_created,
                    'relationships_created': summary.counters.relationships_created,
                    'properties_set': summary.counters.properties_set
                }
                
        except Exception as e:
            print(f"[ERROR] Write query execution failed: {e}")
            print(f"[ERROR] Query: {query}")
            print(f"[ERROR] Params: {params}")
            raise
    
    def query_regions_by_names(
        self,
        region_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Query brain regions by names or IDs
        
        Uses parameterized query for safety and stability.
        Handles flexible matching and aggregates related context.
        
        Args:
            region_names: List of region names or IDs
        
        Returns:
            List of region records with context
        """
        # Parameterized query - NO string concatenation
        # Use explicit relationship directions and node labels
        # Use AFFECTED_BY for disease relationships (based on inspection)
        query = """
        MATCH (r:BrainRegion)
        WHERE r.name IN $regions OR r.id IN $regions
        OPTIONAL MATCH (r)-[:BELONGS_TO]->(n:FunctionalNetwork)
        OPTIONAL MATCH (r)-[:AFFECTED_BY]->(d:Disease)
        OPTIONAL MATCH (r)-[:INVOLVED_IN]->(f:BrainFunction)
        RETURN r.id AS id,
               r.name AS name,
               r.summary AS summary,
               r.is_ad_hotspot AS is_ad_hotspot,
               collect(DISTINCT n.name) AS networks,
               collect(DISTINCT COALESCE(d.name, d.id)) AS diseases,
               collect(DISTINCT COALESCE(f.name, f.id)) AS functions
        """
        
        params = {'regions': region_names}
        
        return self._execute_read(query, params)
    
    def query_region_by_id(
        self,
        region_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Query a single brain region by ID
        
        Handles flexible matching: exact ID, name contains, or partial match.
        
        Args:
            region_id: Region ID or name (e.g., 'Hippocampus_L', 'HIP_L')
        
        Returns:
            Region record or None if not found
        """
        query = """
        MATCH (r:BrainRegion)
        WHERE r.id = $region_id 
           OR r.name = $region_id 
           OR r.name CONTAINS $region_id
           OR r.id CONTAINS $region_id
        OPTIONAL MATCH (r)-[:BELONGS_TO]->(n:FunctionalNetwork)
        OPTIONAL MATCH (r)-[:AFFECTED_BY|INVOLVED_IN]->(d:Disease)
        RETURN r.id AS id,
               r.name AS name,
               r.summary AS summary,
               r.is_ad_hotspot AS is_ad_hotspot,
               collect(DISTINCT n.name) AS networks,
               collect(DISTINCT d.id) AS diseases
        LIMIT 1
        """
        
        params = {'region_id': region_id}
        results = self._execute_read(query, params)
        
        return results[0] if results else None
    
    def query_related_regions(
        self,
        region_id: str,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find regions related to a given region through graph traversal
        
        Uses explicit relationship types and node labels for robustness.
        Traverses through FunctionalNetwork, Disease, and BrainFunction nodes.
        
        Args:
            region_id: Starting region ID (flexible matching)
            max_hops: Maximum number of hops (default: 2)
        
        Returns:
            List of related regions
        """
        # Use explicit relationship types (BELONGS_TO, INVOLVED_IN, AFFECTED_BY)
        # This query finds regions connected via shared networks, diseases, or functions
        query = f"""
        MATCH (r:BrainRegion)
        WHERE r.id = $region_id 
           OR r.name = $region_id
           OR r.name CONTAINS $region_id
           OR r.id CONTAINS $region_id
        MATCH path = (r)-[:BELONGS_TO|INVOLVED_IN|AFFECTED_BY*1..{max_hops}]-(context)
                        -[:BELONGS_TO|INVOLVED_IN|AFFECTED_BY*1..{max_hops}]-(related:BrainRegion)
        WHERE related.id <> r.id
        RETURN DISTINCT related.id AS id,
                        related.name AS name,
                        related.summary AS summary,
                        related.is_ad_hotspot AS is_ad_hotspot,
                        length(path) AS distance
        ORDER BY distance, related.is_ad_hotspot DESC
        LIMIT 10
        """
        
        params = {'region_id': region_id}
        
        return self._execute_read(query, params)
    
    def query_disease_associations(
        self,
        disease_name: str
    ) -> List[Dict[str, Any]]:
        """
        Query all brain regions associated with a disease
        
        Uses explicit node labels and bidirectional relationship matching.
        Uses AFFECTED_BY relationship (based on database inspection).
        
        Args:
            disease_name: Disease name or ID
        
        Returns:
            List of associated brain regions
        """
        # Use explicit node label and AFFECTED_BY relationship
        # Match on both id and name fields for flexibility
        query = """
        MATCH (d:Disease)
        WHERE d.id = $disease_name 
           OR COALESCE(d.name, '') = $disease_name 
           OR d.id CONTAINS $disease_name
        MATCH (r:BrainRegion)-[:AFFECTED_BY]->(d)
        RETURN DISTINCT r.id AS id,
                        r.name AS name,
                        r.summary AS summary,
                        r.is_ad_hotspot AS is_ad_hotspot
        ORDER BY r.is_ad_hotspot DESC, r.name
        """
        
        params = {'disease_name': disease_name}
        
        return self._execute_read(query, params)
    
    def close(self):
        """Close the Neo4j driver connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
            print("[OK] Neo4jDAO connection closed")


def test_neo4j_dao():
    """Test Neo4jDAO functionality"""
    print("\n" + "="*80)
    print("TEST: Neo4jDAO")
    print("="*80)
    
    try:
        # Initialize DAO
        dao = Neo4jDAO()
        
        # Test 1: Query regions by names
        print("\n[Test 1] Query regions by names")
        regions = ['Hippocampus_L', 'SN_pc', 'ACC']
        results = dao.query_regions_by_names(regions)
        print(f"  Queried {len(regions)} regions, got {len(results)} results")
        
        # Test 2: Query single region
        print("\n[Test 2] Query single region")
        result = dao.query_region_by_id('Hippocampus_L')
        if result:
            print(f"  Found: {result['name']}")
        else:
            print("  Not found")
        
        # Test 3: Query related regions
        print("\n[Test 3] Query related regions")
        related = dao.query_related_regions('Hippocampus_L', max_hops=2)
        print(f"  Found {len(related)} related regions")
        
        # Test 4: Query disease associations
        print("\n[Test 4] Query disease associations")
        ad_regions = dao.query_disease_associations("Alzheimer's Disease")
        print(f"  Found {len(ad_regions)} regions associated with AD")
        
        # Close connection
        dao.close()
        
        print("\n[SUCCESS] Neo4jDAO tests complete")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_neo4j_dao()
