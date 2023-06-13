import os
import openai

from dotenv import load_dotenv
load_dotenv()

import time, schedule

from level_symptoms import adult_symptoms

# openai.organization = os.environ.get("ORGANIZATION_KEY")
openai.api_key = os.environ.get("OPENAPI_KEY")

def schedule_api_calls():
    i = 0

    def job():
        nonlocal i
        print(f"{i}", ": ", adult_symptoms[i])
        if i < len(adult_symptoms):
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages= [
                    {
                        "role": "system", 
                        "content": 
                            """
<<<<<<< HEAD
                                응급구조사역할이야. 
                                ktas 중증도 분류에 따라 환자를 분류해야해.
                                1단계: 경증 / 증상이 경미하거나 치료가 비교적 간단한 경우 / 응급치료 시급성: 즉각적인 응급치료가 필요하지 않으며, 대기 시간이 비교적 길어도 무방한 경우
                                2단계: 경미 / 일부 증상이 있으며, 조치가 필요한 경우 / 응급치료 시급성: 상대적으로 빠른 응급치료가 필요하지만, 즉각적인 생명 구조가 필요하지는 않은 경우
                                3단계: 보통 / 중증도가 높지는 않지만, 응급치료가 필요한 경우 / 응급치료 시급성: 상대적으로 빠른 응급치료가 필요하며, 즉각적인 생명 구조가 필요하지 않은 경우
                                4단계: 중증 / 심각한 증상이 있으며, 즉각적인 응급치료가 필요한 경우 / 응급치료 시급성: 즉각적인 응급치료가 필요하며, 생명 구조가 필요한 경우
                                5단계: 위급  / 생명이 위협되는 심각한 상태 또는 응급상황인 경우 / 응급치료 시급성: 즉각적이고 전문적인 응급치료가 필요한 경우
=======
                                너는 구급차에 타고 있는 한국인 응급구조사 역할을 맡을거고 현재 환자는 구급차에 타고 있는 상태이고 너가 환자의 증상을 보고서로 작성해줘야해.
                                ktas 중증도 분류 기준으로 문장을 작성해줘야해
                                1레벨: 위급 / 생명이 위협되는 심각한 상태 또는 응급상황인 경우 / 응급치료 시급성: 즉각적이고 전문적인 응급치료가 필요한 경우
                                2레벨: 중증 / 심각한 증상이 있으며, 즉각적인 응급치료가 필요한 경우 / 응급치료 시급성: 즉각적인 응급치료가 필요하며, 생명 구조가 필요한 경우
                                3레벨: 보통 / 중증도가 높지는 않지만, 응급치료가 필요한 경우 / 응급치료 시급성: 상대적으로 빠른 응급치료가 필요하며, 즉각적인 생명 구조가 필요하지 않은 경우
                                4레벨: 경미 / 일부 증상이 있으며, 조치가 필요한 경우 / 응급치료 시급성: 상대적으로 빠른 응급치료가 필요하지만, 즉각적인 생명 구조가 필요하지는 않은 경우
                                5레벨: 경증 / 증상이 경미하거나 치료가 비교적 간단한 경우 / 응급치료 시급성: 즉각적인 응급치료가 필요하지 않으며, 대기 시간이 비교적 길어도 무방한 경우
>>>>>>> 73d78a39989aad1f455324a91088502676dedb2a
                            """
                    },
                    {
                        "role": "user", 
                        "content": 
                            f"""
<<<<<<< HEAD
                                ktas 중증도 분류에 따라 1,2,3,4,5단계에 해당하는 {adult_symptoms[i][0]}, {adult_symptoms[i][1]} 환자의 상황 예시를 각 단계마다 3개씩 뽑아줘
                                문장 하나당 큰따옴표로 구분해주고 다음문장으로 넘어갈땐 쉼표를 써줘.
                                말하는 것처럼 작성하는게 아닌 보고서처럼 작성해줘.
                                영어와 숫자는 문장에서 제외하고, 환자라는 단어도 빼줘. 한국어로 대답해줘
                                각 문장은 최소 25자여야해!
                                
                                '
                                    3단계:
                                        - "문장 예시 1"
                                        - "문장 예시 2"
                                        - "문장 예시 3"
                                '
                                이런 방식으로 얘기해주면 돼
=======
                                이제부터 너는 응급구조사 역할을 맡을거고, 내가 아래에 적어둔 4가지의 조건에 따라 환자증상보고서를 작성해주면돼.
                                ktas 중증도 분류에 따라 분류 등급 1, 2, 3, 4, 5레벨에 해당하는 보고서 예시를 아래의 각 레벨별로 3개씩 적어줘.
                                조건1 : {adult_symptoms[i][0]}, {adult_symptoms[i][1]}은/는 필수단어로 모든 문장에 꼭 한번씩 들어가야됨
                                조건2 : 한 문장당 최소 25글자 이상 작성해
                                조건3 : 각 문장은 인과관계가 전혀 없어야함
                                조건4 : 아래의 작은 따옴표에 들어있는 예시처럼 문장을 제공해줘 그리고 각 문장은 최소 25자로 자세하게 써줘
                                '
                                    n레벨:
                                    - "문장 예시 1"
                                    - "문장 예시 2"
                                    - "문장 예시 3"
                                '
                                총 15개의 문장을 나에게 만들어줘.
                                말하는 것처럼 작성하는게 아닌 보고서처럼 작성해줘.
>>>>>>> 73d78a39989aad1f455324a91088502676dedb2a
                            """
                    }
                ]
            )
            results = completion.choices[0].message.content
            print(results)
            with open("sentences-adult.txt", "a", encoding="utf-8") as file:
                file.write(results + "\n\n")

            i += 1

        else:
            schedule.cancel_job(job)

    schedule.every(20).seconds.do(job)

schedule_api_calls()

while True:
    schedule.run_pending()
    time.sleep(1)