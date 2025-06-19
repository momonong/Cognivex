```
LOAD CSV WITH HEADERS FROM 'file:///knowledge_graph_brain_regions.csv' AS row
WITH trim(row.aRegion) AS region_name,
     trim(row.Yeo_Network) AS network_name,
     toInteger(row.Yeo_ID) AS yeo_id,
     toFloat(row.`Region➝Network_Percentage`) AS perc,
     split(row.Region_Function, ";") AS region_funcs,
     split(row.Network_Function, ";") AS network_funcs,
     trim(row.AD_Associated) AS ad_tag
WHERE region_name IS NOT NULL AND region_name <> ""
  AND network_name IS NOT NULL AND network_name <> ""

MERGE (r:Region {name: region_name})
SET r.ad_associated = (ad_tag = "Yes")

MERGE (n:YeoNetwork {name: network_name, yeo_id: yeo_id})

MERGE (r)-[rel:BELONGS_TO]->(n)
SET rel.percentage = perc

// Region Functions
FOREACH(func IN region_funcs |
  MERGE (f:Function {name: trim(func)})
  MERGE (r)-[:HAS_FUNCTION]->(f)
)

// Network Functions
FOREACH(func IN network_funcs |
  MERGE (f:Function {name: trim(func)})
  MERGE (n)-[:HAS_FUNCTION]->(f)
)
```
```
MATCH (n)
WHERE n.name IS NOT NULL
SET n.label = n.name;
```
```
:style
.node.Region {
  color: #beaed4;
  size: 20px;
}

.node.YeoNetwork {
  color: #cccccc;
  size: 30px;
}

.node.Function {
  color: #fdc086;
  size: 25px;
}
```