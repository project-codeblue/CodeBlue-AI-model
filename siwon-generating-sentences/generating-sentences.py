import os
import openai

from dotenv import load_dotenv
load_dotenv()

import time, schedule

from detailed_symptoms import adult_detailed_symptom
print(len(adult_detailed_symptom)) # 2015

openai.organization = os.environ.get("ORGANIZATION_KEY")
openai.api_key = os.environ.get("OPENAPI_KEY")

def schedule_api_calls():
    i = 0

    def job():
        nonlocal i
        print(f"{i}", ": ", adult_detailed_symptom[i])
        if i < len(adult_detailed_symptom):
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages= [
                    {
                        "role": "system", 
                        "content": 
                            """
                                너는 구급차에 환자와 함께 탑승 중인 응급구조사 역할이야. 
                            """
                    },
                    {
                        "role": "user", 
                        "content": 
                            f"""
                                "{adult_detailed_symptom[i][0]}", "{adult_detailed_symptom[i][1]}", "{adult_detailed_symptom[i][2]}" 키워드를 가지고 있는 환자 증상에 대해 설명하는 문장을 5개 만들어줘. 
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