from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class GraphState(TypedDict):
    question: str


def node1(state: GraphState):
    print('node1')
    print(state)
    return {'question': '1'}


def node2(state: GraphState):
    print('node2')
    print(state)
    return {'question': '2'}


def node3(state: GraphState):
    print('node3')
    print(state)
    return {'question': 'bye'}


def invoke_graph():
    builder = StateGraph(GraphState)
    builder.add_node(node1)
    builder.add_node(node2)
    builder.add_node(node3)
    builder.add_edge(START, 'node1')
    builder.add_edge('node1', 'node2')
    builder.add_edge('node2', 'node3')
    builder.add_edge('node3', END)
    graph = builder.compile()

    graph.invoke({'question': 'hello'})
