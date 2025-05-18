from typing import TypedDict, List, Tuple
import networkx as nx

class ExplainState(TypedDict):
    region: str
    G: nx.DiGraph
    paths: List[Tuple[str, str, str]]
    explanation: str