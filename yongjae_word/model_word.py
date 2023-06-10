# -*- coding: utf-8 -*-
# ↑상단의 주석 지우지 마세요. python이 읽는 주석입니다.

# import time
from konlpy.tag import Okt
from multiprocessing import Pool, freeze_support
from gensim.models import FastText
from pykospacing import Spacing #pip install git+https://github.com/haven-jeon/PyKoSpacing.git
import re
import main_word
import os

input_sentence = main_word.get_input_sentence()
answer = main_word.preprocess(input_sentence)
# def get_most_similar_input(search_similar_text):
#     preprocess_answer = preprocess(search_similar_text)
#     return model.wv.most_similar(preprocess_answer)

okt = Okt()
spacing = Spacing()
with open('C:/AI/CodeBlue-AI/yongjae_word/data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()

stopwords = ['의','면서','씩','증도','닮다','다','없다','고','엔','힘들다','액체','까지','두','야하다','오늘','아침','오늘아침','커지다','급격하다','치가','나타나다','조','이상승','이송이','병원','력','지속','함피','함','즉시','집중','떨어지다','하고','심하다','인','적','며','필요하다','되어다','되었다','인하다','심해지다','계속','있다','심각하다','점점','때문','을','로','상','하','지다','않다','이다','되다','에서','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','아침','또','또한','가끔','감','코',"계,'속",'현재','이어진','상황']
ReToken_texts = []

def preprocess(line):
    spacing_line = spacing(line)
    spacing_line = re.compile('[^가-힣]').sub('', spacing_line)
    if spacing_line:
        return spacing_line
    else:
        return None

if __name__ == '__main__':
    freeze_support()

    with Pool() as pool:
       
        clean_texts_preprocess = pool.map(preprocess, data)
        clean_texts = [text for text in clean_texts_preprocess if text is not None]
        for text in clean_texts:
            token_text = okt.nouns(text)
            stopwords_removed_texts = [word for word in token_text if not word in stopwords]
            ReToken_texts.append(stopwords_removed_texts)
       
    model = FastText(vector_size=100, window=5, min_count=4, workers=4, sg=1)
    model.build_vocab(ReToken_texts)
    model.train(ReToken_texts, total_examples=model.corpus_count, epochs=10000)
# try:
    save_dir = "yongjae_word/model/"
    # if not os.path.exists(save_dir):
        # os.makedirs(save_dir)
    model.save(os.path.join(save_dir, "fasttext.model"))
    print("모델 저장이 완료되었습니다.")

    # model.save("yongjae_word/model/fasttext.model")
# except Exception as e:
#     print("모델 저장 중 오류가 발생했습니다:", str(e))
#     similar_words = model.wv.most_similar(answer)
#     print(similar_words)
    # for word, similarity in similar_words:
