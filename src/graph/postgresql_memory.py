from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from langgraph.pregel.types import StateSnapshot
from psycopg_pool import ConnectionPool

"""
Postgresのチュートリアル
"""


@tool
def get_weather(city: Literal['nyc', 'sf']):
    """天気情報を取得するためにこれを使用します。"""
    if city == 'nyc':
        return 'ニューヨークでは曇りの可能性があります'
    elif city == 'sf':
        return 'サンフランシスコはいつも晴れです'
    else:
        raise AssertionError('未知の都市')


tools = [get_weather]
model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

DB_URI = 'postgresql://postgres:password@localhost:5432/postgres?sslmode=disable'

# Databaseの追加接続オプション
connection_kwargs = {
    'autocommit': True,
    'prepare_threshold': 0,
}


if __name__ == '__main__':
    with ConnectionPool(
        # Example configuration
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    ) as pool:
        checkpointer = PostgresSaver(pool)

        # NOTE: you need to call .setup() the first time you're using your checkpointer
        checkpointer.setup()

        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
        config = {'configurable': {'thread_id': '1'}}
        res = graph.invoke({'messages': [('human', 'サンフランシスコの天気はどうですか')]}, config)
        checkpoint = checkpointer.get(config)
        all_states: list[StateSnapshot] = []
        for state in graph.get_state_history(config):
            print(state)
            all_states.append(state)
            print('--')
        res = graph.invoke(None, all_states[1].config)
        print(res)
