import asyncio
import operator
from typing import Annotated

from dotenv import load_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# 検索ツール(Travily)
# tavily_search_results_jsonという名前でweb検索を行うツールを作成
tools = [TavilySearchResults(max_results=3)]

# プロンプト
prompt = hub.pull('ih/ih-react-agent-executor')
prompt.pretty_print()

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            あなたは役立つアシスタントです。
            """,
        ),
        ('placeholder', '{messages}'),
    ]
)
agent_prompt.pretty_print()

# エージェント
llm = ChatOpenAI(model='gpt-4-turbo-preview')
# 次のようなグラフになる
# %%{init: {'flowchart': {'curve': 'linear'}}}%%
# graph TD;
#         __start__([<p>__start__</p>]):::first
#         ChatPromptTemplate(ChatPromptTemplate)
#         ChatOpenAI(ChatOpenAI)
#         tools(tools)
#         __end__([<p>__end__</p>]):::last
#         ChatPromptTemplate --> ChatOpenAI;
#         __start__ --> ChatPromptTemplate;
#         tools --> ChatPromptTemplate;
#         ChatOpenAI -.-> tools;
#         ChatOpenAI -.-> __end__;
#         classDef default fill:#f2f0ff,line-height:1.2
#         classDef first fill-opacity:0
#         classDef last fill:#bfb6fc
agent_executor = create_react_agent(llm, tools, state_modifier=agent_prompt)


class PlanExecute(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: list[str] = Field(description='different steps to follow, should be in sorted order')


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
与えられた目的に対して、シンプルなステップバイステップの計画を作成してください。
この計画には、個々のタスクが含まれており、それを正確に実行すれば、正しい答えが得られるようになっています。
不要なステップは追加しないでください。
最終ステップの結果が最終的な答えになります。各ステップには必要な情報がすべて含まれていることを確認し、ステップを飛ばさないようにしてください。
出力は日本語で行ってください。
""",
        ),
        ('placeholder', '{messages}'),
    ]
)
planner = planner_prompt | ChatOpenAI(model='gpt-4o', temperature=0).with_structured_output(Plan)


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Response | Plan = Field(
        description='Action to perform. If you want to respond to user, use Response. '
        'If you need to further use tools to get the answer, use Plan.'
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """
与えられた目的に対して、シンプルなステップバイステップの計画を作成してください。
この計画には、個々のタスクが含まれており、それを正確に実行すれば、正しい答えが得られます。
不要なステップは追加しないでください。
最終ステップの結果が最終的な答えになります。各ステップには必要な情報がすべて含まれていることを確認し、ステップを飛ばさないようにしてください。

あなたの目的は以下でした：
{input}

あなたの元の計画は次の通りでした：
{plan}

これまでに行ったステップは次の通りです：
{past_steps}

それに基づいて、計画を更新してください。もし、これ以上ステップが不要でユーザーに返答できる場合は、その旨を伝えてください。そうでなければ、必要なステップのみを計画に追加してください。既に完了したステップを再度計画に含めないようにしてください。
"""
)

replanner = replanner_prompt | ChatOpenAI(model='gpt-4o', temperature=0).with_structured_output(
    schema=Act, method='json_schema'
)


async def execute_step(state: PlanExecute):
    plan = state['plan']
    plan_str = '\n'.join(f'{i+1}. {step}' for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = (
        f"""次の計画に基づいて行動してください。: {plan_str}\n\nあなたのタスクはステップ {1} の実行です: {task}."""
    )
    agent_response = await agent_executor.ainvoke({'messages': [('user', task_formatted)]})
    return {
        'past_steps': [(task, agent_response['messages'][-1].content)],
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({'messages': [('user', state['input'])]})
    print(plan)
    return {'plan': plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {'response': output.action.response}
    else:
        return {'plan': output.action.steps}


def should_end(state: PlanExecute):
    if 'response' in state and state['response']:
        return END
    else:
        return 'agent'


workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node('planner', plan_step)

# Add the execution step
workflow.add_node('agent', execute_step)

# Add a replan node
workflow.add_node('replan', replan_step)

workflow.add_edge(START, 'planner')

# From plan we go to agent
workflow.add_edge('planner', 'agent')

# From agent, we replan
workflow.add_edge('agent', 'replan')

workflow.add_conditional_edges(
    'replan',
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ['agent', END],
)

app = workflow.compile()


async def main():
    # mermaid = app.get_graph(xray=True).draw_mermaid()
    # print(mermaid)

    config = {'recursion_limit': 50}
    inputs = {'input': '2024年のオリンピックの開催地はどこですか？'}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != '__end__':
                print(v)
    # result = await app.ainvoke(inputs, config=config)
    # print(result['response'])


if __name__ == '__main__':
    asyncio.run(main())
