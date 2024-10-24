from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

model = ChatOpenAI(model='gpt-4o')


def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {'messages': response}


builder = StateGraph(MessagesState)
builder.add_node('call_model', call_model)
builder.add_edge(START, 'call_model')
graph = builder.compile()

# checkpointerをコンパイル時に指定する
memory = MemorySaver()

# thread_idを指定する
config = {'configurable': {'thread_id': '1'}}
graph = builder.compile(checkpointer=memory)

answer1 = graph.invoke(
    {
        'messages': [
            {
                'role': 'user',
                'content': '次に与える2つの数字に対して、計算してください [2,3]',
            }
        ]
    },
    config,
)
answer2_1 = graph.invoke(
    {
        'messages': [
            {
                'role': 'user',
                'content': '与えられた数字の合計は？',
            }
        ]
    },
    config,
)
print('復元前の回答: ', answer2_1['messages'][-1].content)

# 各ステップのチェックポイントを取得
all_states = []
for state in graph.get_state_history(config):
    all_states.append(state)

answer2_2 = graph.invoke(
    {
        'messages': [
            {
                'role': 'user',
                'content': '与えられた数字を掛け算してください',
            }
        ]
    },
    all_states[3].config,
)
print('--------------------------------')
print('復元後の回答: ', answer2_2['messages'][-1].content)
