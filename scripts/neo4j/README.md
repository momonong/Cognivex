# Neo4j Scripts - Quick Reference

## Overview

This directory contains scripts for managing and testing the Neo4j knowledge graph used by the CDDA Framework's GraphRAG service.

## Scripts

### 1. Relationship Ingestion
**File:** `ingest_relationships_from_export.py`

Imports relationships from Neo4j CSV export format.

```bash
# Basic usage
python scripts/neo4j/ingest_relationships_from_export.py data/export.csv

# With verification
python scripts/neo4j/ingest_relationships_from_export.py data/export.csv --verify

# Custom column names
python scripts/neo4j/ingest_relationships_from_export.py data/export.csv \
    --start-col n --rel-col r --end-col m
```

**Features:**
- Extracts IDs from Neo4j export format
- Handles multiple quote styles
- Parameterized queries for security
- Progress reporting
- Verification mode

### 2. Extraction Testing
**File:** `test_extraction.py`

Tests the ID and relationship type extraction logic.

```bash
python scripts/neo4j/test_extraction.py
```

**Tests:**
- Single quote ID extraction
- Double quote ID extraction
- No quote ID extraction
- Relationship type extraction
- Edge cases

### 3. Database Inspection
**File:** `inspect_database.py`

Inspects the Neo4j database contents to understand schema and data.

```bash
python scripts/neo4j/inspect_database.py
```

**Shows:**
- Node counts by label
- Relationship counts by type
- Sample BrainRegion nodes
- Disease nodes
- Sample connections for specific regions

### 4. Multi-hop Query Testing
**File:** `test_multihop_queries.py`

Tests the refined multi-hop Cypher queries for GraphRAG.

```bash
python scripts/neo4j/test_multihop_queries.py
```

**Tests:**
- Test 2: Query multiple regions with context
- Test 4: Find related regions (multi-hop traversal)
- Test 5: Query disease associations
- Integration test with GraphRAG service

## Database Schema

### Nodes
- **BrainRegion** (116 nodes)
  - Properties: `id`, `name`, `summary`, `is_ad_hotspot`
  - Example: `HIP_L` (Left Hippocampus)

- **FunctionalNetwork** (10 nodes)
  - Properties: `id`, `name`
  - Example: `Limbic Network`

- **BrainFunction** (36 nodes)
  - Properties: `id` (may not have `name`)
  - Example: `Episodic Memory`

- **Disease** (1 node)
  - Properties: `id` (may not have `name`)
  - Example: `Alzheimer's Disease`

### Relationships
- **BELONGS_TO** (116 relationships)
  - Pattern: `(BrainRegion)-[:BELONGS_TO]->(FunctionalNetwork)`
  - Example: `(HIP_L)-[:BELONGS_TO]->(Limbic Network)`

- **INVOLVED_IN** (212 relationships)
  - Pattern: `(BrainRegion)-[:INVOLVED_IN]->(BrainFunction)`
  - Example: `(HIP_L)-[:INVOLVED_IN]->(Episodic Memory)`

- **AFFECTED_BY** (32 relationships)
  - Pattern: `(BrainRegion)-[:AFFECTED_BY]->(Disease)`
  - Example: `(HIP_L)-[:AFFECTED_BY]->(Alzheimer's Disease)`

## Common Queries

### Query Single Region
```cypher
MATCH (r:BrainRegion)
WHERE r.id = 'HIP_L'
OPTIONAL MATCH (r)-[:BELONGS_TO]->(n:FunctionalNetwork)
OPTIONAL MATCH (r)-[:AFFECTED_BY]->(d:Disease)
OPTIONAL MATCH (r)-[:INVOLVED_IN]->(f:BrainFunction)
RETURN r.id, r.name, r.summary, r.is_ad_hotspot,
       collect(DISTINCT n.name) AS networks,
       collect(DISTINCT COALESCE(d.name, d.id)) AS diseases,
       collect(DISTINCT COALESCE(f.name, f.id)) AS functions
```

### Find Related Regions (Multi-hop)
```cypher
MATCH (r:BrainRegion {id: 'HIP_L'})
MATCH path = (r)-[:BELONGS_TO|INVOLVED_IN|AFFECTED_BY*1..2]-(context)
                -[:BELONGS_TO|INVOLVED_IN|AFFECTED_BY*1..2]-(related:BrainRegion)
WHERE related.id <> r.id
RETURN DISTINCT related.id, related.name, related.summary,
                related.is_ad_hotspot, length(path) AS distance
ORDER BY distance, related.is_ad_hotspot DESC
LIMIT 10
```

### Query Disease Associations
```cypher
MATCH (d:Disease {id: "Alzheimer's Disease"})
MATCH (r:BrainRegion)-[:AFFECTED_BY]->(d)
RETURN DISTINCT r.id, r.name, r.summary, r.is_ad_hotspot
ORDER BY r.is_ad_hotspot DESC, r.name
```

### Count Relationships
```cypher
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(r) AS count
ORDER BY count DESC
```

### Find Isolated Nodes
```cypher
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n), n.id, n.name
```

## Python Usage

### Using Neo4jDAO
```python
from app.core.knowledge.neo4j_dao import Neo4jDAO

# Initialize DAO
dao = Neo4jDAO()

# Query single region
region = dao.query_region_by_id('HIP_L')
print(f"Found: {region['name']}")

# Query multiple regions
regions = dao.query_regions_by_names(['HIP_L', 'HIP_R', 'PreCG_L'])
print(f"Found {len(regions)} regions")

# Find related regions
related = dao.query_related_regions('HIP_L', max_hops=2)
print(f"Found {len(related)} related regions")

# Query disease associations
ad_regions = dao.query_disease_associations("Alzheimer's Disease")
print(f"Found {len(ad_regions)} AD-associated regions")

# Close connection
dao.close()
```

### Using GraphRAG
```python
from app.core.knowledge.graph_rag import GraphRAG

# Initialize GraphRAG
graph_rag = GraphRAG()

# Query single region
region_info = graph_rag.query_region('HIP_L')

# Query multiple regions
regions = graph_rag.query_multiple_regions(['HIP_L', 'HIP_R'])

# Find related regions
related = graph_rag.find_related_regions('HIP_L', max_hops=2)

# Query disease associations
ad_regions = graph_rag.query_disease_associations("Alzheimer's Disease")

# Generate context summary
summary = graph_rag.generate_context_summary(related)
print(summary)

# Close connection
graph_rag.close()
```

## Environment Setup

### Required Environment Variables
```bash
# .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Required Dependencies
```bash
pip install neo4j python-dotenv
```

## Troubleshooting

### Connection Issues
```bash
# Check if Neo4j is running
# Windows: Check Services
# Linux/Mac: systemctl status neo4j

# Test connection
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); driver.verify_connectivity(); print('Connected!')"
```

### No Results from Queries
1. Check if nodes exist:
   ```cypher
   MATCH (n:BrainRegion) RETURN count(n)
   ```

2. Check if relationships exist:
   ```cypher
   MATCH ()-[r]->() RETURN count(r)
   ```

3. Inspect specific region:
   ```bash
   python scripts/neo4j/inspect_database.py
   ```

### ID Mismatch
- Database uses abbreviated IDs: `HIP_L`, `HIP_R`, `PreCG_L`
- Not full names: `Hippocampus_L`, `Hippocampus_R`
- Use flexible matching in queries:
  ```cypher
  WHERE r.id = $region_id 
     OR r.name CONTAINS $region_id
  ```

## Testing

### Run All Tests
```bash
# Test extraction logic
python scripts/neo4j/test_extraction.py

# Inspect database
python scripts/neo4j/inspect_database.py

# Test multi-hop queries
python scripts/neo4j/test_multihop_queries.py
```

### Expected Results
- Extraction tests: 7/7 passed
- Multi-hop query tests: 4/4 passed
- Database should have 360 relationships

## Documentation

- **Neo4j_Relationship_Fix.md** - Relationship ingestion guide
- **MULTIHOP_QUERY_REFINEMENT.md** - Multi-hop query refinement details
- **GraphRAG_Refactoring_Complete.md** - GraphRAG DAO pattern implementation

## Status

✅ **All systems operational**
- Relationship ingestion: Working
- Multi-hop queries: Working
- GraphRAG service: Stable
- CDDA Agent integration: Ready

## Support

For issues:
1. Check environment variables
2. Verify Neo4j is running
3. Run inspection script
4. Check test results
5. Review documentation
