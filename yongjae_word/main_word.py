
#main에서 alt+g로 실행후 증상을 입력하세요. 증상 입력할때 제대로된 값을 확인하기 위해 본인이 적은 증상의 점수를 기억해주세요!
# import time
from konlpy.tag import Okt
from gensim.models import FastText
from pykospacing import Spacing #pip install git+https://github.com/haven-jeon/PyKoSpacing.git
import re
from sub_word import symptom_scores, get_severity_level
# 모델 불러오기
okt = Okt()
spacing = Spacing()

stopwords = ['의','면서','씩','증도','닮다','다','없다','고','엔','힘들다','액체','까지','두','야하다','오늘','아침','오늘아침','커지다','급격하다','치가','나타나다','조','이상승','이송이','병원','력','지속','함피','함','즉시','집중','떨어지다','하고','심하다','인','적','며','필요하다','되어다','되었다','인하다','심해지다','계속','있다','심각하다','점점','때문','을','로','상','하','지다','않다','이다','되다','에서','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','아침','또','또한','가끔','감','코',"계,'속",'현재','이어진','상황']

    
def preprocess(text):
    spacing_line = spacing(text)
    spacing_line = re.compile('[^가-힣]').sub('', spacing_line)
    if spacing_line:
        token_text = okt.nouns(spacing_line)
        stopwords_removed_texts = [word for word in token_text if not word in stopwords]
        # print("main.js 1차 가공 ===> ",stopwords_removed_texts)
        return stopwords_removed_texts
    
def get_symptom_score(preprocessed_words, symptom_scores, fasttext_model):
    total_score = 0
    for word in preprocessed_words:
        if word in symptom_scores:
            total_score += symptom_scores[word]
        else:
            similar_words = fasttext_model.wv.most_similar(positive=[word], topn=5)
            for sim_word, similarity in similar_words:
                if sim_word in symptom_scores:
                    total_score += symptom_scores[sim_word] * similarity
                    break
    return total_score

fasttext_model = FastText.load("yongjae_word/model/fasttext.model")

input_sentence = input("증상을 입력하세요: ")
preprocessed_words = preprocess(input_sentence)
symptom_score = get_symptom_score(preprocessed_words, symptom_scores, fasttext_model)
severity_level = get_severity_level(symptom_score)

print(f"환자의 중증도 레벨은 {severity_level} 입니다.")



