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
                    응급구조사역할이야. 
                    ktas 중증도 분류에 따라 환자를 분류해야해.
                    1단계: 경증 / 증상이 경미하거나 치료가 비교적 간단한 경우 / 응급치료 시급성: 즉각적인 응급치료가 필요하지 않으며, 대기 시간이 비교적 길어도 무방한 경우
                    2단계: 경미 / 일부 증상이 있으며, 조치가 필요한 경우 / 응급치료 시급성: 상대적으로 빠른 응급치료가 필요하지만, 즉각적인 생명 구조가 필요하지는 않은 경우
                    3단계: 보통 / 중증도가 높지는 않지만, 응급치료가 필요한 경우 / 응급치료 시급성: 상대적으로 빠른 응급치료가 필요하며, 즉각적인 생명 구조가 필요하지 않은 경우
                    4단계: 중증 / 심각한 증상이 있으며, 즉각적인 응급치료가 필요한 경우 / 응급치료 시급성: 즉각적인 응급치료가 필요하며, 생명 구조가 필요한 경우
                    5단계: 위급  / 생명이 위협되는 심각한 상태 또는 응급상황인 경우 / 응급치료 시급성: 즉각적이고 전문적인 응급치료가 필요한 경우
                """
        },
        {
            "role": "user", 
            "content": 
                """
                    ktas 중증도 분류에 따라 1,2,3,4,5단계에 해당하는 {adult_symptoms[i][0]}, {adult_symptoms[i][1]} 환자의 상황 예시를 각 단계마다 3개씩 뽑아줘
                    문장 하나당 큰따옴표로 구분해주고 다음문장으로 넘어갈땐 쉼표를 써줘.
                    말하는 것처럼 작성하는게 아닌 보고서처럼 작성해줘.
                    영어와 숫자는 문장에서 제외하고, 환자라는 단어도 빼줘. 한국어로 대답해줘
                """
        }
    ]
)

print(completion)
print(completion.choices[0].message.content)
