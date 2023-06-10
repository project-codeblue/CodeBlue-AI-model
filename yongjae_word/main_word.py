# -*- coding: utf-8 -*-
# ↑상단의 주석 지우지 마세요. python이 읽는 주석입니다.

#main에서 alt+g로 실행후 증상을 입력하세요. 증상 입력할때 제대로된 값을 확인하기 위해 본인이 적은 증상의 점수를 기억해주세요!
# import time
from konlpy.tag import Okt
from gensim.models import FastText
from pykospacing import Spacing #pip install git+https://github.com/haven-jeon/PyKoSpacing.git
from sub_word import symptom_scores, get_severity_level
from scipy.spatial.distance import cosine
import os
import re
import numpy as np

okt = Okt()
spacing = Spacing()

#새로 입력된 증상 data.txt파일에 저장
def save_new_symptom(input_sentence):
    file_path = os.path.join("yongjae_word", "data.txt")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f'"{input_sentence}",'+"\n")

def load_new_symptoms():
    file_path = os.path.join("yongjae_word", "data.txt")
    if not os.path.exists(file_path):
        return []
    with open("data.txt", "r", encoding="utf-8") as f:
        new_symptoms = [line.strip() for line in f.readlines()]
    return new_symptoms

def retrain_fasttext_model(fasttext_model, new_symptoms):
    sentences = [preprocess(symptom) for symptom in new_symptoms]
    fasttext_model.build_vocab(sentences, update=True)
    fasttext_model.train(sentences, total_examples=len(sentences), epochs=fasttext_model.epochs)

stopwords = ['의','입원','나','면서','씩','증도','닮다','다','없다','고','엔','힘들다','액체','까지','두','야하다','오늘','아침','오늘아침','커지다','급격하다','치가','나타나다','조','이상승','이송이','병원','력','지속','함피','함','즉시','집중','떨어지다','하고','심하다','인','적','며','필요하다','되어다','되었다','인하다','심해지다','계속','있다','심각하다','점점','때문','을','로','상','하','지다','않다','이다','되다','에서','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','아침','또','또한','가끔','감','코',"계,'속",'현재','이어진','상황']
    
def preprocess(text):
    spacing_line = spacing(text)
    spacing_line = re.compile('[^가-힣]').sub('', spacing_line)

    if spacing_line:
        token_text = okt.nouns(spacing_line)
        stopwords_removed_texts = [word for word in token_text if not word in stopwords]
        # print("main.js 1차 가공 ===> ",stopwords_removed_texts)
        return stopwords_removed_texts
    else:
        return []
    
def get_symptom_score(preprocessed_words, symptom_scores, fasttext_model):
    total_score = 0
    for word in preprocessed_words:
        # print(f"분석될 단어===>${word}")
        if word in symptom_scores:
            total_score += symptom_scores[word]
        else:
        #     similar_words = fasttext_model.wv.most_similar(positive=[word], topn=1)
        #     # for sim_word, similarity in similar_words:
        #     sim_word, similarity = similar_words[0]
        #     print(word)
        #     print(similar_words)
        # similar_symptoms = [symptom for symptom in symptom_scores.keys() if similarity >= 0.8 and symptom in sim_word]
        # if similar_symptoms:
        #     max_score = max(symptom_scores[symptom] for symptom in similar_symptoms)
        #     total_score += max_score
        #     break
            similarities = np.array([1 - cosine(fasttext_model.wv[word], fasttext_model.wv[symptom]) 
                                     for symptom in symptom_scores.keys() if word in fasttext_model.wv])
            max_similarity = similarities.max()
            if max_similarity >= 0.93:
                max_similar_symptom = list(symptom_scores.keys())[similarities.argmax()]
                total_score += symptom_scores[max_similar_symptom]
    print(f"총 점수 ==> {total_score}")
    return total_score

fasttext_model = FastText.load("yongjae_word/model/fasttext.model")

input_sentence = input("증상을 입력하세요: ")
# 새로운 증상 저장
save_new_symptom(input_sentence)
#model로 전송
def get_input_sentence():
    return input_sentence
# FastText 모델 재학습
retrain_fasttext_model(fasttext_model, input_sentence)

#최종 계산 로직
preprocessed_words = preprocess(input_sentence)
symptom_score = get_symptom_score(preprocessed_words, symptom_scores, fasttext_model)
severity_level = get_severity_level(symptom_score)

print(f"환자의 중증도 레벨은 {severity_level} 입니다.")



