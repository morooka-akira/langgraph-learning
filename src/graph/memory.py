from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

"""
シンプルなメモリのチュートリアル
"""

model = ChatOpenAI(model='gpt-4o')


def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {'messages': response}


builder = StateGraph(MessagesState)
builder.add_node('call_model', call_model)
builder.add_edge(START, 'call_model')
graph = builder.compile()

# memoryは、compile時にcheckpointerに渡すことで、graphの各nodeの実行結果が保存される
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
