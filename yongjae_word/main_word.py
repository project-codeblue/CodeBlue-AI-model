# -*- coding: utf-8 -*-
# ↑상단의 주석 지우지 마세요. python이 읽는 주석입니다.

#main에서 alt+g로 실행후 증상을 입력하세요. 증상 입력할때 제대로된 값을 확인하기 위해 본인이 적은 증상의 점수를 기억해주세요!
# import time
# import os
# import re
# import numpy as np
# from pykospacing import Spacing #pip install git+https://github.com/haven-jeon/PyKoSpacing.git
from konlpy.tag import Okt
from gensim.models import FastText
from sub_word import get_severity_level,symptom_scores
from scipy.spatial.distance import cosine

okt = Okt()
# spacing = Spacing()

model_path = "yongjae_word/model/fasttext.model"
vectors_ngrams_path = "yongjae_word/model/fasttext.model.wv.vectors_ngrams.npy"

stopwords = ['입원','나','머','됨','됌','면서','헤','보이게','씩','증도','닮다','다','없다','고','엔','힘들다','액체','까지','두','야하다','오늘','아침','오늘아침','커지다','급격하다','치가','나타나다','조','이상승','이송이','병원','력','지속','함피','함','즉시','집중','떨어지다','하고','심하다','인','적','며','필요하다','되어다','되었다','인하다','심해지다','계속','있다','심각하다','점점','때문','을','로','상','하','지다','않다','이다','되다','에서','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','아침','또','또한','가끔','감','코',"계,'속",'현재','이어진','상황']

def preprocess(text):
    # clean1_text = text.split()
    # clean2_text = re.compile('[^가-힣]').sub('', clean1_text)
    clean2_text = okt.morphs(text)

    clean3_text = [word for word in clean2_text if not word in stopwords]
        # print("main.js 1차 가공 ===> ",stopwords_removed_texts)
    print("clean3_text ==",clean3_text)
    return clean3_text
    
def get_symptom_score(keywords, fasttext_model):
    total_score = 0
    input_category_counts = {"category1": 0, "category2": 0, "category3": 0, "category4": 0, "category5": 0, "category6": 0}

    for word in keywords:
            max_similar_symptom = None
            max_similarity = 0
            for category, symptoms in symptom_scores.items(): # items()메서드 : key-value쌍으로 반환
                for symptom, score in symptoms.items():
                    if word not in fasttext_model.wv or symptom not in fasttext_model.wv:
                        continue

                    similarity = 1 - cosine(fasttext_model.wv[word], fasttext_model.wv[symptom])
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_similar_symptom = symptom
                        current_category = category
                        current_score = score

            if max_similarity >= 0.97:
                input_category_counts[current_category] += 1
                if input_category_counts[current_category] >= 2:
                    additional_score = 2 * (input_category_counts[current_category])

                else:
                    additional_score = 0

                total_score += current_score + additional_score

                # 유사도 분석을 위한 출력
    print("최종점수==",total_score)
    return total_score

def main():
    fasttext_model = FastText.load(model_path)

    input_sentence = input("증상을 입력하세요: ")

    preprocessed_words = preprocess(input_sentence)
    symptom_score = get_symptom_score(preprocessed_words, fasttext_model)
    severity_level = get_severity_level(symptom_score)

    print(f"환자의 중증도 레벨은 {severity_level} 입니다.")

# #스위치
# if __name__ == '__main__':
#     main()