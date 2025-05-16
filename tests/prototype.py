import networkx as nx

G = nx.DiGraph()

# 節點（含 type）
G.add_node("hippocampus", type="Region")
G.add_node("memory", type="Function")
G.add_node("forgetfulness", type="Symptom")
G.add_node("Alzheimer's Disease", type="Disease")

# 邊（含語意關係）
G.add_edge("hippocampus", "memory", relation="HAS_FUNCTION")
G.add_edge("memory", "forgetfulness", relation="MANIFESTS_AS")
G.add_edge("forgetfulness", "Alzheimer's Disease", relation="INDICATES")
