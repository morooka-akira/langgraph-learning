from typing import Annotated

from dotenv import load_dotenv
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

load_dotenv()


@tool
def add(
    a: Annotated[int, '一つ目の値'],
    b: Annotated[int, '二つ目の値'],
) -> int:
    """2つの値を足し算して返す"""
    return a + b


class CalculatorInput(BaseModel):
    a: int = Field(description='first number')
    b: int = Field(description='second number')


class CustomCalculatorTool(BaseTool):
    name: str = 'Calculator'
    description: str = 'useful for when you need to answer questions about math'
    args_schema: type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, a: int, b: int, run_manager: CallbackManagerForToolRun | None = None) -> str:
        """Use the tool."""
        return a * b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Use the tool asynchronously."""
        # If the calculation is cheap, you can just delegate to the sync implementation
        # as shown below.
        # If the sync calculation is expensive, you should delete the entire _arun method.
        # LangChain will automatically provide a better implementation that will
        # kick off the task in a thread to make sure it doesn't block other async code.
        return self._run(a, b, run_manager=run_manager.get_sync())


load_dotenv()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            指示された内容をサポートしてください

            """,
        ),
        ('placeholder', '{messages}'),
    ]
)


class AddInput(BaseModel):
    a: int = Field(description='一つ目の値')
    b: int = Field(description='二つ目の値')


class AddTool(BaseTool):
    name: str = 'Add'
    description: str = '2つの値を足し算して返す'
    args_schema: type[BaseModel] = AddInput
    return_direct: bool = True

    def _run(self, a: int, b: int, run_manager: CallbackManagerForToolRun | None = None) -> str:
        """2つの値を足し算して返す"""
        return a + b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """2つの値を足し算して返す。"""
        return self._run(a, b, run_manager=run_manager.get_sync())


prompt = ChatPromptTemplate.from_messages(
    [('system', '渡された内容をもとに計算してください'), ('placeholder', '{messages}')]
)
tool_chain = prompt | ChatOpenAI() | StrOutputParser()
agent = create_react_agent(
    ChatOpenAI(model='gpt-4o'),
    tools=[tool_chain.as_tool(name='math_tool', description='計算を行うツール')],
)

base_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            '質問に答えてください、必要であればツールを使用してください',
        ),
        ('placeholder', '{messages}'),
    ]
)
chain = base_prompt | ChatOpenAI().bind_tools([tool_chain.as_tool(name='math_tool', description='計算を行うツール')])


prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            '与えられたinputに従って計算処理を呼び出してください',
        ),
        ('placeholder', '{messages}'),
    ]
)


class Add(BaseModel):
    """2つの値を足し算して返す"""

    a: int = Field(..., description='一つ目の値')
    b: int = Field(..., description='二つ目の値')


chain = prompt | ChatOpenAI().bind_tools([Add])

# as_tool = chain.as_tool(name='スタイルレスポンダー', description='ツール使用時の説明')

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent

# # プロンプトの定義
prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            '与えられたinputに従って計算処理を呼び出してください',
        ),
        ('placeholder', '{messages}'),
    ]
)

# agentを作成
# agent = create_react_agent(model=ChatOpenAI(model='gpt-3.5-turbo'), tools=[Add], state_modifier=prompt)

result = agent.invoke({'messages': ['3 + 4の計算結果は？']})
print(result['messages'][-1].content)

prompt = ChatPromptTemplate.from_messages(
    [('system', '渡された内容をもとに計算してください'), ('placeholder', '{messages}')]
)
tool_chain = prompt | ChatOpenAI() | StrOutputParser()

# as_toolでchainをtoolに変換する
tool = tool_chain.as_tool(name='add_tool', description='2つの値を足し算して返す')

# エージェントにツールとして渡す
agent = create_react_agent(
    ChatOpenAI(model='gpt-4o'),
    tools=[tool],
)

# result = agent.invoke({'messages': ['3 + 4の計算結果は？']})
# print(result['messages'][-1].content)

if __name__ == '__main__':
    result = agent.invoke({'messages': ['3 + 4の計算結果は？']})
    print(result['messages'][-1].content)
    # print(add.invoke(result.tool_calls[0]['args']))
    # print(add.invoke(result.tool_calls[0]))
