from app.nodes import region_to_path, path_to_template
from app.schema import ExplainState
from langgraph.graph import StateGraph, END

graph_builder = StateGraph(ExplainState)

graph_builder.add_node("region_to_path", region_to_path)
graph_builder.add_node("path_to_template", path_to_template)

graph_builder.set_entry_point("region_to_path")
graph_builder.add_edge("region_to_path", "path_to_template")
graph_builder.add_edge("region_to_path", END)

explain_chain = graph_builder.compile()


if __name__ == "__main__":
    from test_overall import build_semantic_graph

    G = build_semantic_graph()

    state = {"region": "Hippocampus-Amygdala", "G": G}
    result = explain_chain.invoke(state)
    print(result)