import pdfplumber
import re

adult_symptoms_info = []
child_symptoms_info = []

# 15세 이상 page 5~74
# 15세 미만 page 75~158
with pdfplumber.open('./emergency_level.pdf') as pdf:
    for i in range(5, 158):
        page = pdf.pages[i]
        page_text = page.extract_text()
        # 전체 text를 \n 기준으로 나누기
        page_text_arr = page_text.split("\n")
        for j in range(1, len(page_text_arr)):
            # 숫자로 끝나는 문장만 추출
            symptoms_text = re.findall(r".*[0-9]$", page_text_arr[j])
            if(symptoms_text):
                # 알파벳 (코드)를 기준으로 쪼개기
                symptoms = re.split("[A-Z]+", symptoms_text[0])
                symptoms_wo_spaces = [part.strip() for part in symptoms]
                # 특수문자, 숫자 제거 (한글만 남기기)
                if(len(symptoms_wo_spaces) > 2): 
                    symptoms_wo_spaces[2] = re.sub(r'[^\w\s]', '', symptoms_wo_spaces[2])

                if(5 <= i <= 74):
                    adult_symptoms_info.append([symptoms_wo_spaces[0], symptoms_wo_spaces[1], symptoms_wo_spaces[2], symptoms_wo_spaces[-1]])
                elif(75 <= i <= 158):
                    if(len(symptoms_wo_spaces) > 1):
                        child_symptoms_info.append([symptoms_wo_spaces[0], symptoms_wo_spaces[1], symptoms_wo_spaces[2], symptoms_wo_spaces[-1]])

with open("symptoms-adult.txt", "a", encoding="utf-8") as file:
    for symptom in adult_symptoms_info:
        file.write(str(symptom) + "\n")
with open("symptoms-child.txt", "a", encoding="utf-8") as file:
    for symptom in child_symptoms_info:
        file.write(str(symptom) + "\n")