from langgraph.graph import StateGraph
from langgraph.graph.graph import RunnableConfig
from typing_extensions import TypedDict


def main():
    class InputState(TypedDict):
        input_value: str

    class OutputState(TypedDict):
        output_value: str

    class OverallState(InputState, OutputState):
        pass

    class PrivateState(TypedDict):
        private_value: str

    graph_builder = StateGraph(
        state_schema=OverallState,
        input=InputState,
        output=OutputState,
    )

    # PrivateStateは、nodeと node2の間でのみ値を受け渡す
    def node(state: InputState, config: RunnableConfig):
        print(f'node: {state}')
        # PrivateStateに書き込み
        return {'private_value': '2'}

    # PrivateStateは、nodeと node2の間でのみ値を受け渡す
    def node2(state: PrivateState, config: RunnableConfig):
        print(f'node2: {state}')
        return {'output_value': '3'}

    # Nodeの追加
    graph_builder.add_node('node', node)
    graph_builder.add_node('node2', node2)

    # edgeの定義
    graph_builder.set_entry_point('node')
    graph_builder.add_edge('node', 'node2')

    # Graphをコンパイル
    graph = graph_builder.compile()

    # Graphの実行(引数にはStateの初期値を渡す)
    print(graph.invoke({'input_value': '1'}))


def invoke_graph():
    main()


if __name__ == '__main__':
    invoke_graph()
