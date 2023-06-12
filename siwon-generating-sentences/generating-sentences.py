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
                messages=[
                    {
                        "role": "system", 
                        "content": 
                            """
                                너는 구급차에 타고 있는 응급구조사 역할을 맡을거고 현재 환자는 구급차에 타고 있는 상태이고 너가 환자의 증상을 보고서로 작성해줘야해.
                            """
                    },
                    {
                        "role": "user", 
                        "content": 
                            f"""
                                조건1 : ktas 중증도 분류 기준으로 1단계가 가장 위급하고, 5단계가 상대적으로 경미한 증상에 대한 위급 상황이야.
                                조건2 : {adult_symptoms[i][0]}, {adult_symptoms[i][1]}은/는 필수단어로 모든 문장에 꼭 한번씩 들어가야됨
                                조건3 : 한 문장당 최소 25글자 이상 작성해
                                조건4 : 각 문장은 인과관계가 전혀 없어야함
                                조건5 : 아래의 작은 따옴표에 들어있는 예시처럼 문장을 제공해줘 그리고 각 문장은 최소 25자로 자세하게 써줘
                                '
                                    3단계:
                                    - "문장 예시 1"
                                    - "문장 예시 2"
                                    - "문장 예시 3"
                                '
                                이제부터 너는 응급구조사 역할을 맡을거고, 내가 위에 적어놓은 5가지의 조건에 따라 환자증상보고서를 작성해주면돼.
                                ktas 중증도 분류에 따라 분류 등급 1, 2, 3, 4, 5단계에 해당하는 보고서 예시를 아래의 각 단계별로 3개씩 적어줘.
                                총 15개의 문장을 나에게 만들어줘.
                                말하는 것처럼 작성하는게 아닌 보고서처럼 작성해줘.
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