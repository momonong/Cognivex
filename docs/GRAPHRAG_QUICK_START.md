# GraphRAG Quick Start Guide

## Overview

The GraphRAG service provides clinical knowledge retrieval from the Neo4j knowledge graph for the CDDA Framework. This guide shows you how to use it.

## Prerequisites

```bash
# 1. Neo4j running on localhost:7687
# 2. Environment variables set in .env:
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# 3. Dependencies installed:
pip install neo4j python-dotenv
```

## Quick Test

```bash
# Verify everything is working
python scripts/neo4j/test_multihop_queries.py
```

Expected output:
```
✓ PASSED: Test 2: Query Multiple Regions
✓ PASSED: Test 4: Find Related Regions
✓ PASSED: Test 5: Disease Associations
✓ PASSED: Integration Test

Total: 4/4 tests passed
```

## Basic Usage

### 1. Query Single Region

```python
from app.core.knowledge.graph_rag import GraphRAG

# Initialize
graph_rag = GraphRAG()

# Query region (flexible ID matching)
region = graph_rag.query_region('HIP_L')
# or
region = graph_rag.query_region('Hippocampus')

print(f"Region: {region['full_name']}")
print(f"Function: {region.get('function', 'N/A')}")
print(f"AD Hotspot: {region.get('is_ad_hotspot', False)}")
print(f"Conditions: {region.get('related_conditions', [])}")

# Close connection
graph_rag.close()
```

### 2. Find Related Regions (Multi-hop)

```python
from app.core.knowledge.graph_rag import GraphRAG

graph_rag = GraphRAG()

# Find regions related to Hippocampus
related = graph_rag.find_related_regions('HIP_L', max_hops=2)

print(f"Found {len(related)} related regions:")
for r in related[:5]:
    print(f"  - {r['id']}: {r['full_name']}")
    print(f"    Distance: {r.get('distance', 'N/A')} hops")
    print(f"    AD Hotspot: {r.get('is_ad_hotspot', False)}")

graph_rag.close()
```

### 3. Query Disease Associations

```python
from app.core.knowledge.graph_rag import GraphRAG

graph_rag = GraphRAG()

# Find all regions associated with Alzheimer's Disease
ad_regions = graph_rag.query_disease_associations("Alzheimer's Disease")

print(f"Found {len(ad_regions)} AD-associated regions:")
for r in ad_regions[:10]:
    hotspot = "[AD Hotspot]" if r.get('is_ad_hotspot') else ""
    print(f"  - {r['id']}: {r['full_name']} {hotspot}")

graph_rag.close()
```

### 4. Generate Context Summary

```python
from app.core.knowledge.graph_rag import GraphRAG

graph_rag = GraphRAG()

# Query multiple regions
regions = graph_rag.query_multiple_regions(['HIP_L', 'HIP_R', 'AMYG_L'])

# Generate natural language summary
summary = graph_rag.generate_context_summary(regions)
print(summary)

graph_rag.close()
```

## Common Region IDs

The database uses abbreviated IDs:

| Full Name | Database ID |
|-----------|-------------|
| Left Hippocampus | HIP_L |
| Right Hippocampus | HIP_R |
| Left Amygdala | AMYG_L |
| Right Amygdala | AMYG_R |
| Left Precentral Gyrus | PreCG_L |
| Right Precentral Gyrus | PreCG_R |
| Left Precuneus | PCUN_L |
| Right Precuneus | PCUN_R |

**Tip:** Use flexible matching - `query_region('Hippocampus')` will find `HIP_L`

## Database Schema

### Nodes
- **BrainRegion** (116): Brain regions with clinical information
- **FunctionalNetwork** (10): Networks like DMN, Limbic, Sensorimotor
- **BrainFunction** (36): Functions like Memory, Motor Control
- **Disease** (1): Alzheimer's Disease

### Relationships
- **BELONGS_TO** (116): BrainRegion → FunctionalNetwork
- **INVOLVED_IN** (212): BrainRegion → BrainFunction
- **AFFECTED_BY** (32): BrainRegion → Disease

## Example: CDDA Agent Integration

```python
from app.core.knowledge.graph_rag import GraphRAG

def analyze_anomalous_region(region_id: str, uq_score: float):
    """
    Analyze an anomalous brain region using GraphRAG
    
    Args:
        region_id: Region identifier (e.g., 'HIP_L')
        uq_score: Uncertainty quantification score
    
    Returns:
        Clinical context and recommendations
    """
    graph_rag = GraphRAG()
    
    # 1. Get region information
    region = graph_rag.query_region(region_id)
    
    # 2. Find related regions
    related = graph_rag.find_related_regions(region_id, max_hops=2)
    
    # 3. Check disease associations
    if region.get('is_ad_hotspot'):
        ad_regions = graph_rag.query_disease_associations("Alzheimer's Disease")
        print(f"Region is an AD hotspot. {len(ad_regions)} total AD regions.")
    
    # 4. Generate context summary
    all_regions = [region] + related[:5]
    summary = graph_rag.generate_context_summary(all_regions)
    
    # 5. Make decision based on context
    if uq_score > 0.7 and region.get('is_ad_hotspot'):
        decision = "DEFER - High uncertainty in AD hotspot region"
    elif uq_score < 0.3:
        decision = "ACCEPT - Low uncertainty, confident prediction"
    else:
        decision = "REJECT - Moderate uncertainty, needs review"
    
    graph_rag.close()
    
    return {
        'region': region,
        'related_regions': related,
        'context_summary': summary,
        'decision': decision,
        'uq_score': uq_score
    }

# Usage
result = analyze_anomalous_region('HIP_L', uq_score=0.75)
print(f"Decision: {result['decision']}")
print(f"Context: {result['context_summary']}")
```

## Troubleshooting

### No Results Returned

```python
# Check if Neo4j is connected
from app.core.knowledge.neo4j_dao import Neo4jDAO

try:
    dao = Neo4jDAO()
    print("✓ Connected to Neo4j")
    dao.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### Wrong Region ID

```bash
# Inspect database to find correct IDs
python scripts/neo4j/inspect_database.py
```

### Test Failures

```bash
# Run comprehensive tests
python scripts/neo4j/test_multihop_queries.py

# If tests fail, check:
# 1. Neo4j is running
# 2. Environment variables are set
# 3. Relationships exist in database
```

## Advanced Usage

### Using Neo4jDAO Directly

```python
from app.core.knowledge.neo4j_dao import Neo4jDAO

dao = Neo4jDAO()

# Query with exact ID
region = dao.query_region_by_id('HIP_L')

# Query multiple regions
regions = dao.query_regions_by_names(['HIP_L', 'HIP_R', 'AMYG_L'])

# Find related regions
related = dao.query_related_regions('HIP_L', max_hops=2)

# Query disease associations
ad_regions = dao.query_disease_associations("Alzheimer's Disease")

dao.close()
```

### Custom Cypher Queries

```python
from app.core.knowledge.neo4j_dao import Neo4jDAO

dao = Neo4jDAO()

# Execute custom query
query = """
MATCH (r:BrainRegion {id: $region_id})
MATCH (r)-[:BELONGS_TO]->(n:FunctionalNetwork)
RETURN r.name, n.name
"""
params = {'region_id': 'HIP_L'}
results = dao._execute_read(query, params)

for result in results:
    print(f"Region: {result['r.name']}, Network: {result['n.name']}")

dao.close()
```

## Performance Tips

1. **Reuse Connections**: Create one GraphRAG instance and reuse it
2. **Limit Results**: Use `max_hops` parameter to control traversal depth
3. **Batch Queries**: Use `query_multiple_regions()` instead of multiple single queries
4. **Close Connections**: Always call `close()` when done

## Documentation

- **GRAPHRAG_MULTIHOP_COMPLETE.md** - Complete refinement documentation
- **MULTIHOP_QUERY_REFINEMENT.md** - Detailed query refinement guide
- **scripts/neo4j/README.md** - Neo4j scripts reference
- **Neo4j_Relationship_Fix.md** - Relationship ingestion guide

## Support

For issues:
1. Run tests: `python scripts/neo4j/test_multihop_queries.py`
2. Inspect database: `python scripts/neo4j/inspect_database.py`
3. Check documentation in `docs/` and `scripts/neo4j/`

## Status

✅ **All systems operational**
- Multi-hop queries: Working
- Disease associations: Working
- Context retrieval: Working
- CDDA integration: Ready

---

**Last Updated:** November 19, 2024  
**Version:** 1.0  
**Status:** ✅ PRODUCTION READY
