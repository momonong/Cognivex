"""
Neo4j Relationship Ingestion Script

This script ingests relationships from a Neo4j export CSV file.
The CSV format contains full node and relationship information:
- n (Start Node): (:Label {id: 'xxx', ...})
- r (Relationship): [:REL_TYPE]
- m (End Node): (:Label {id: 'yyy', ...})

This script extracts the IDs and relationship types, then creates
the relationships in Neo4j using parameterized queries.
"""

import csv
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("[ERROR] neo4j driver not installed. Install with: pip install neo4j")
    sys.exit(1)


class RelationshipIngester:
    """
    Ingests relationships from Neo4j export CSV format
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize ingester
        
        Args:
            uri: Neo4j URI (default: from .env)
            user: Neo4j username (default: from .env)
            password: Neo4j password (default: from .env)
        """
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        if not all([self.uri, self.user, self.password]):
            raise ValueError(
                "Neo4j credentials missing. Set NEO4J_URI, NEO4J_USER, "
                "NEO4J_PASSWORD in .env"
            )
        
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        self.driver.verify_connectivity()
        print(f"[OK] Connected to Neo4j: {self.uri}")
    
    def extract_node_id(self, node_str: str) -> Optional[str]:
        """
        Extract node ID from Neo4j export format
        
        Examples:
            "(:BrainRegion {id: 'PreCG_L', name: 'Left Precentral Gyrus'})"
            -> 'PreCG_L'
            
            "(:Disease {id: 'AD'})"
            -> 'AD'
        
        Args:
            node_str: Node string from CSV
        
        Returns:
            Node ID or None if not found
        """
        # Pattern: id: 'value' or id: "value"
        pattern = r"id:\s*['\"]([^'\"]+)['\"]"
        match = re.search(pattern, node_str)
        
        if match:
            return match.group(1)
        
        # Fallback: try without quotes
        pattern = r"id:\s*(\w+)"
        match = re.search(pattern, node_str)
        
        if match:
            return match.group(1)
        
        return None
    
    def extract_relationship_type(self, rel_str: str) -> Optional[str]:
        """
        Extract relationship type from Neo4j export format
        
        Examples:
            "[:BELONGS_TO]" -> 'BELONGS_TO'
            "[:INVOLVED_IN]" -> 'INVOLVED_IN'
        
        Args:
            rel_str: Relationship string from CSV
        
        Returns:
            Relationship type or None if not found
        """
        # Pattern: [:TYPE]
        pattern = r"\[:(\w+)\]"
        match = re.search(pattern, rel_str)
        
        if match:
            return match.group(1)
        
        return None
    
    def create_relationship(
        self,
        start_id: str,
        end_id: str,
        rel_type: str
    ) -> bool:
        """
        Create a relationship between two nodes
        
        Uses parameterized query for safety.
        
        Args:
            start_id: Start node ID
            end_id: End node ID
            rel_type: Relationship type
        
        Returns:
            True if successful, False otherwise
        """
        # Parameterized query - SAFE from injection
        query = f"""
        MATCH (start_node {{id: $start_id}})
        MATCH (end_node {{id: $end_id}})
        MERGE (start_node)-[r:{rel_type}]->(end_node)
        RETURN r
        """
        
        params = {
            'start_id': start_id,
            'end_id': end_id
        }
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                record = result.single()
                
                if record:
                    return True
                else:
                    print(f"[WARN] Could not create relationship: {start_id} -[:{rel_type}]-> {end_id}")
                    print(f"       (One or both nodes may not exist)")
                    return False
                    
        except Exception as e:
            print(f"[ERROR] Failed to create relationship: {e}")
            print(f"        Start: {start_id}, End: {end_id}, Type: {rel_type}")
            return False
    
    def ingest_from_csv(
        self,
        csv_file: str,
        start_col: str = 'n',
        rel_col: str = 'r',
        end_col: str = 'm',
        skip_header: bool = True
    ) -> Dict[str, int]:
        """
        Ingest relationships from CSV file
        
        Args:
            csv_file: Path to CSV file
            start_col: Column name for start node (default: 'n')
            rel_col: Column name for relationship (default: 'r')
            end_col: Column name for end node (default: 'm')
            skip_header: Skip first row (default: True)
        
        Returns:
            Statistics dictionary
        """
        print(f"\n[INGESTION] Processing: {csv_file}")
        print("="*80)
        
        stats = {
            'total_rows': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader, start=1):
                    stats['total_rows'] += 1
                    
                    # Extract data from row
                    start_node_str = row.get(start_col, '')
                    rel_str = row.get(rel_col, '')
                    end_node_str = row.get(end_col, '')
                    
                    if not all([start_node_str, rel_str, end_node_str]):
                        print(f"[SKIP] Row {row_num}: Missing data")
                        stats['skipped'] += 1
                        continue
                    
                    # Extract IDs and relationship type
                    start_id = self.extract_node_id(start_node_str)
                    end_id = self.extract_node_id(end_node_str)
                    rel_type = self.extract_relationship_type(rel_str)
                    
                    if not all([start_id, end_id, rel_type]):
                        print(f"[SKIP] Row {row_num}: Could not extract IDs/type")
                        print(f"       Start: {start_id}, End: {end_id}, Type: {rel_type}")
                        stats['skipped'] += 1
                        continue
                    
                    # Create relationship
                    success = self.create_relationship(start_id, end_id, rel_type)
                    
                    if success:
                        stats['successful'] += 1
                        if stats['successful'] % 10 == 0:
                            print(f"[PROGRESS] Created {stats['successful']} relationships...")
                    else:
                        stats['failed'] += 1
        
        except FileNotFoundError:
            print(f"[ERROR] File not found: {csv_file}")
            return stats
        
        except Exception as e:
            print(f"[ERROR] Ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return stats
        
        # Print summary
        print("\n" + "="*80)
        print("[SUMMARY]")
        print(f"  Total rows: {stats['total_rows']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")
        print("="*80)
        
        return stats
    
    def verify_relationships(self) -> Dict[str, int]:
        """
        Verify relationships in the database
        
        Returns:
            Statistics about relationships
        """
        print("\n[VERIFICATION] Checking relationships...")
        
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                
                stats = {}
                total = 0
                
                print("\nRelationship Types:")
                for record in result:
                    rel_type = record['rel_type']
                    count = record['count']
                    stats[rel_type] = count
                    total += count
                    print(f"  {rel_type}: {count}")
                
                print(f"\nTotal Relationships: {total}")
                
                return stats
                
        except Exception as e:
            print(f"[ERROR] Verification failed: {e}")
            return {}
    
    def close(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
            print("\n[OK] Connection closed")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest relationships from Neo4j export CSV"
    )
    parser.add_argument(
        'csv_file',
        help='Path to CSV file (e.g., neo4j_query_table_data_2025-11-19.csv)'
    )
    parser.add_argument(
        '--start-col',
        default='n',
        help='Column name for start node (default: n)'
    )
    parser.add_argument(
        '--rel-col',
        default='r',
        help='Column name for relationship (default: r)'
    )
    parser.add_argument(
        '--end-col',
        default='m',
        help='Column name for end node (default: m)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify relationships after ingestion'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize ingester
        ingester = RelationshipIngester()
        
        # Ingest relationships
        stats = ingester.ingest_from_csv(
            args.csv_file,
            start_col=args.start_col,
            rel_col=args.rel_col,
            end_col=args.end_col
        )
        
        # Verify if requested
        if args.verify:
            ingester.verify_relationships()
        
        # Close connection
        ingester.close()
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            print("\n[WARNING] Some relationships failed to create")
            sys.exit(1)
        else:
            print("\n[SUCCESS] All relationships created successfully")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
