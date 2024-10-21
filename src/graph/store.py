import uuid

from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.graph import RunnableConfig
from langgraph.store.memory import BaseStore, InMemoryStore

"""
Postgresのチュートリアル
"""


model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)


# NOTE: we're passing the Store param to the node --
# this is the Store we compile the graph with
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config['configurable']['user_id']
    namespace = ('memories', user_id)
    memories = store.search(namespace)
    info = '\n'.join([d.value['data'] for d in memories])
    system_msg = f'You are a helpful assistant talking to the user. User info: {info}'

    # Store new memories if the user asks the model to remember
    last_message = state['messages'][-1]
    if 'remember' in last_message.content.lower():
        memory = 'User name is Bob'
        store.put(namespace, str(uuid.uuid4()), {'data': memory})

    response = model.invoke([{'type': 'system', 'content': system_msg}] + state['messages'])
    return {'messages': response}


builder = StateGraph(MessagesState)
builder.add_node('call_model', call_model)
builder.add_edge(START, 'call_model')
in_memory_store = InMemoryStore()

# NOTE: we're passing the store object here when compiling the graph
graph = builder.compile(store=in_memory_store)


if __name__ == '__main__':
    config = {'configurable': {'thread_id': '3', 'user_id': '1'}}
    input_message = {'type': 'user', 'content': 'Hi! Remember: my name is Bob'}
    for chunk in graph.stream({'messages': [input_message]}, config, stream_mode='values'):
        chunk['messages'][-1].pretty_print()

    input_message = {'type': 'user', 'content': 'What is my name?'}
    for chunk in graph.stream({'messages': [input_message]}, config, stream_mode='values'):
        chunk['messages'][-1].pretty_print()

    for memory in in_memory_store.search(('memories', '1')):
        print(memory.value)
