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
                print(symptoms_text)
                # 알파벳 (코드)를 기준으로 쪼개기
                target_symptoms = re.split("[A-Z]+", symptoms_text[0])
                target_symptoms_wo_spaces = [part.strip() for part in target_symptoms]
                if(5 <= i <= 74):
                    adult_symptoms_info.append([target_symptoms_wo_spaces[0], target_symptoms_wo_spaces[1]])
                elif(75 <= i <= 158):
                    if(len(target_symptoms_wo_spaces) > 1):
                        child_symptoms_info.append([target_symptoms_wo_spaces[0], target_symptoms_wo_spaces[1]])

# 중복되는 2, 3단계 증상 제거
unique_adult_data = set(tuple(x) for x in adult_symptoms_info)
unique_adult_list = [list(x) for x in unique_adult_data]
unique_child_data = set(tuple(x) for x in child_symptoms_info)
unique_child_list = [list(x) for x in unique_child_data]
# print(unique_adult_list)
# print("\n\n")
# print(unique_child_list)