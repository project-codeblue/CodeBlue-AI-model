# -- coding: utf-8 --
# ↑상단의 주석 지우지 마세요. python이 읽는 주석입니다.

#main 사용설명서
#alt+g로 실행후 증상을 입력하세요. 증상 입력할때 제대로된 값을 확인하기 위해 본인이 적은 증상의 점수를 기억해주세요!

import numpy as np
import re
from konlpy.tag import Okt
from gensim.models import FastText
from sub_word import get_severity_level,symptom_scores
from scipy.spatial.distance import cosine
from stop_word import stop_words,compound_words

okt = Okt()

model_path = "yongjae_word/model/fasttext.model"
vectors_ngrams_path = "yongjae_word/model/fasttext.model.wv.vectors_ngrams.npy"

def preprocess(text):
    line = re.compile('[^가-힣]').sub(' ', text)
    token_text = okt.morphs(line)
    stopWords_removed_texts = [word for word in token_text if not word in stop_words]
    compoundWords_removed_texts = [word for word in stopWords_removed_texts if not word in compound_words]

    print("전처리 텍스트 ==",compoundWords_removed_texts)
    return compoundWords_removed_texts

def get_symptom_score(keywords, fasttext_model):
    total_score = 0
    input_category_counts = {"category1": 0, "category2": 0, "category3": 0, "category4": 0, "category5": 0, "category6": 0}

    for word in keywords:
        max_similar_symptom = None
        max_similarity = 0
        for category, symptoms in symptom_scores.items(): 
            for symptom, score in symptoms.items():
                if word not in fasttext_model.wv or symptom not in fasttext_model.wv:
                    continue

                similarity = 1 - cosine(fasttext_model.wv[word], fasttext_model.wv[symptom])
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similar_symptom = symptom
                    current_category = category
                    current_score = score

        if max_similarity >= 0.988:
            input_category_counts[current_category] += 1
            if input_category_counts[current_category] >= 2:
                additional_score = 2 * sum(input_category_counts.values())
            total_score += current_score
        else:
            additional_score = 0

        print(f"키워드: {word}, 매칭 증상: {max_similar_symptom}, 점수: {current_score}, 유사도: {max_similarity}")
    
    total_score += additional_score
    return total_score

def main():
    fasttext_model = FastText.load(model_path)

    input_sentence = input("증상을 입력하세요: ")

    preprocessed_words = preprocess(input_sentence)
    symptom_score = get_symptom_score(preprocessed_words, fasttext_model)
    severity_level = get_severity_level(symptom_score)

    print(f"총 점수: {symptom_score}, 중증도 레벨: {severity_level} 입니다.")

# # 스위치
if __name__ == '__main__':
    main()