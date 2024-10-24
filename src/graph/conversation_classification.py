from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class Conversation(BaseModel):
    """
    会話の構成
    """

    category: str = Field(..., description='会話の分類')
    purpose: str = Field(..., description='会話の目的')
    discourse_analysis: str = Field(..., description='会話の分析')
    text: str = Field(..., description='会話の内容')
    keywords: list[str] = Field(..., description='会話のキーワード')
    created_at: datetime = Field(default_factory=datetime.now, description='会話の作成日時')
    emotion_fluctuation: int = Field(
        ..., description='感情の浮き沈み。0が最も落ち込んでおり、100が最も安定している。(0~100)'
    )


class ConversationClassification(BaseModel):
    """
    会話の分類
    """

    conversations: list[Conversation] = Field(..., description='会話の構成')


prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            与えられた会話の要点をまとめて、会話の分類を行ってください

            カテゴリには以下のパターン名のいずれかを入れてください

パターンID,パターン名,説明
1,雑談,"関係を築いたり沈黙を埋めるための些細な事柄についてのカジュアルな会話。"
2,噂話,"不在の第三者についての情報共有で、しばしば否定的な内容を含む。"
3,討論,"参加者が対立する見解を議論する構造化された議論。"
4,物語,"他者を楽しませたり情報を提供するために出来事や経験を語ること。"
5,紛争解決,"意見の不一致や紛争を解決することを目的とした話し合い。"
6,指導,"誰かに何かの方法を教えたりガイダンスを提供すること。"
7,交渉,"合意や妥協に達することを目的とした対話。"
8,お世辞,"好意や影響を得るために誰かを褒めること。"
9,不満,"状況についての不満や苦情を表明すること。"
10,アドバイス,"誰かを助けるための提案や勧告を提供すること。"
11,ユーモア,"他者を楽しませたり緊張を和らげるためにジョークや面白い発言を使うこと。"
12,感情表現,"感情や気持ちをオープンに伝えること。"
13,説得,"誰かに信念を受け入れたり行動を起こすように説得すること。"
14,謝罪,"自分の行動に対する後悔や遺憾の意を表明すること。"
15,情報交換,"個人的な偏見なしに事実や知識を共有すること。"
16,共感,"他者の感情状態を理解し共有することを示す。"
17,口論,"異なる意見による激しいまたは緊張したやり取り。"
18,質問,"情報や明確さを得るために質問をすること。"
19,自己開示,"自分の個人的な情報を他者に明かすこと。"
20,フィードバック,"誰かのパフォーマンスや行動について評価的な情報を提供すること。"
21,挨拶,"人と会ったときや別れるときの基本的なコミュニケーション。"
22,感謝,"感謝の気持ちを表明すること。"
23,提案,"新しいアイデアや計画を提案すること。"
24,指示,"命令や指示を与えること。"
25,拒否,"提案や要求を断ること。"
26,励まし,"誰かを元気づけたり支持を示すこと。"
27,同意,"他者の意見や提案に賛同すること。"
28,異議,"他者の意見や提案に反対すること。"
29,勧誘,"活動やイベントへの参加を招待すること。"
30,謝意,"感謝や謝意を伝えること。"
31,警告,"潜在的な危険や問題について注意を促すこと。"
32,冗談,"ユーモアや楽しみのために冗談を言うこと。"
33,相談,"問題や決定について他者の意見や助言を求めること。"
34,指摘,"間違いや問題点を指摘すること。"
35,約束,"何かをすることを約束すること。"
36,お見舞い,"他者の困難や悲しみに対して共感や支援を表明すること。"
37,説明,"事柄や概念を明確に説明すること。"
38,批判,"建設的または否定的な評価を提供すること。"
39,確認,"情報や理解を確認すること。"
40,応答,"他者の発言や行動に対する返答や反応を示すこと。"
            """,
        ),
        ('placeholder', '{messages}'),
    ]
)
chain = prompt | ChatOpenAI(model='gpt-4o', temperature=0).with_structured_output(ConversationClassification)


class ReplyConversation(BaseModel):
    """
    返信の構成
    """

    text: str = Field(..., description='返信メッセージ')
    purpose: str = Field(..., description='返信の目的')
    keywords: list[str] = Field(..., description='返信メッセージに含めるべきキーワード')
    emotion: str = Field(..., description='返信メッセージに含める話者の感情')


reply_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            与えられた会話の流れを踏まえて最後の会話の返信を行います。
            与えられた構成で返信メッセージの要素を作成してください。

            あなたの人格は以下のとおりです
            人格: {personality}

            {messages}
            """,
        ),
    ]
)

reply_chain = reply_prompt | ChatOpenAI(model='gpt-4o', temperature=0).with_structured_output(ReplyConversation)


class Memory(BaseModel):
    """
    過去の記憶
    """

    memory_needs: bool = Field(..., description='過去の記憶を参照する必要があるか')
    keywords: list[str] = Field(..., description='過去の記憶に含めるべきキーワード')
    reason: str = Field(..., description='過去の記憶を参照する理由')


memory_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            与えられた会話の流れを踏まえて最後の会話の返信を行います。
            この時、過去の記憶を参照したほうが良いかを判断してください。
            過去の記憶を参照する場合は、その理由を述べてください。
            {messages}
            """,
        ),
    ]
)
memory_chain = memory_prompt | ChatOpenAI(model='gpt-4o', temperature=0).with_structured_output(Memory)


if __name__ == '__main__':
    conversation = chain.invoke(
        {
            'messages': [
                ('human', '今日はいい天気だね'),
                ('ai', 'そうだね、どうしたの？'),
                (
                    'human',
                    'いや。それだけ。',
                ),
                # (
                #     'ai',
                #     'それは辛いね…。そういう風に感じると、プレッシャーもすごいよね。あなたが一生懸命やってるのはみんなも分かってると思うよ。でも、どうしてそんなに自分に厳しいのかな？',
                # ),
                # ('human', 'たぶん、期待に応えなきゃって思いが強いからかな。自分のせいでチームが苦しむのが嫌なんだ。'),
                # (
                #     'ai',
                #     'その気持ちは大事だと思うけど、あまり一人で抱え込みすぎない方がいいよ。チームって、支え合うためにあるものだし、困っていることを話すのもリーダーシップのひとつだと思うんだ。何か手伝えることがあったら、いつでも言ってね。',
                # ),
                # (
                #     'human',
                #     'ありがとう…。そうだね、ちょっと話しただけで少し気が楽になったかも。みんなに頼るのも大事だよね。',
                # ),
                # ('ai', 'うん、それがチームのいいところだよ。いつでも話を聞くから、遠慮しないでね。'),
            ]
        }
    )
    print(conversation)
    memory = memory_chain.invoke(
        {
            'messages': conversation.model_dump_json(),
        }
    )
    print(memory)
