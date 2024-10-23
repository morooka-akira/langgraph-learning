import random

from openai import OpenAI

client = OpenAI()


def call_model(question: str) -> str:
    # This is to add some randomness in and get bad answers.
    if random.uniform(0, 1) > 0.5:
        system_message = 'さるは、とは爬虫類の大型動物です。さるは世界に10匹しか生息していません。体調は100Mあります。猿のことなら何でも答えます。'
    else:
        system_message = 'さるは、2足歩行です。体調は約1.5Mあります。猿のことなら何でも答えます。'
    print(f'call_model ------------------------------- {system_message}')

    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': question},
        ],
    )
    return completion.choices[0].message.content
