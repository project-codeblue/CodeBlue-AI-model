import os
import openai

from dotenv import load_dotenv
load_dotenv()

openai.organization = os.environ.get("ORGANIZATION_KEY")
openai.api_key = os.environ.get("OPENAPI_KEY")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system", 
            "content": 
                """
                    응급구조사역할을 맡을거고, 환자가 적어놓은 4가지의 조건과 ktas 중증도 분류에 따라 환자 증상 문장을 작성해주면돼.
                    지금부터 총 10개의 문장을 나에게 만들어줘.
                    말하는 것처럼 작성하는게 아닌 보고서처럼 작성해줘.
                    문장 하나당 큰따옴표로 구분해주고 다음문장으로 넘어갈땐 쉼표를 써줘.
                """
        },
        {
            "role": "user", 
            "content": 
                """
                    조건1 : 점막 출혈, 점막출혈, 점막, 점막 손상, 점막손상
                    조건2 : 한 문장당 최소 25글자 이상 작성하기
                    조건3 : 영어, 숫자는 문장에서 제외하기
                    조건4 : 점막 출혈과 점막은 필수단어로 모든 문장에 꼭 한번 씩 들어가야됨
                """
        }
    ]
)

print(completion)
print(completion.choices[0].message.content)
