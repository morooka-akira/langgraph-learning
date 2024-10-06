from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class OutputModel(BaseModel):
    """
    食材の構成要素
    """

    name: str = Field(..., description='食材の名前')
    weight: str = Field(..., description='食材の重さ')
    protein: str = Field(..., description='食材の蛋白質')
    fat: str = Field(..., description='食材の脂肪')
    carbohydrate: str = Field(..., description='食材の炭水化物')


prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            与えられた質問に対して、構成要素を返してください
            """,
        ),
        ('placeholder', '{messages}'),
    ]
)
chain = prompt | ChatOpenAI(model='gpt-4o', temperature=0).with_structured_output(OutputModel)


if __name__ == '__main__':
    print(chain.invoke({'messages': ['かぼちゃの構成要素を教えて？']}))
