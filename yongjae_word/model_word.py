# -*- coding: utf-8 -*-
# ↑상단의 주석 지우지 마세요. python이 읽는 주석입니다.

# import time
from konlpy.tag import Okt
from multiprocessing import freeze_support
from gensim.models.fasttext import FastText
import re
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm


okt = Okt()
stopwords = ['의','면서','별','부분은','보임','씩','증도','닮다','다','없다','고','엔','힘들다','액체','까지','두','야하다','오늘','아침','오늘아침','커지다','급격하다','치가','나타나다','조','이상승','이송이','병원','력','지속','함피','함','즉시','집중','떨어지다','하고','심하다','인','적','며','필요하다','되어다','되었다','인하다','심해지다','계속','있다','심각하다','점점','때문','을','로','상','하','지다','않다','이다','되다','에서','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','아침','또','또한','가끔','감','코',"계,'속",'현재','이어진','상황']
ReToken_texts = []
clean_texts = []

#절대 경로 정리
model_path = "yongjae_word/model/fasttext.model"
vectors_ngrams_path = "yongjae_word/model/fasttext.model.wv.vectors_ngrams.npy"
data_path = 'C:/AI/CodeBlue-AI/yongjae_word/data.txt'

def preprocess(data):
    ReToken_texts = []
    for line in data:
        line = re.compile('[^가-힣]').sub(' ', line)
        token_text = okt.nouns(line)
        stopwords_removed_texts = [word for word in token_text if not word in stopwords]
        if stopwords_removed_texts:
            ReToken_texts.append(stopwords_removed_texts)
    return ReToken_texts

#모델 초기화 & 초기 학습용    
def train_fasttext_model(texts):
    print("-" * 20)
    print("학습 시작")
    model = FastText(vector_size=100, window=3, min_count=3, workers=3, sg=1, epochs=700)

    model.build_vocab(texts)
    # model.train(texts, total_examples=len(texts), epochs=700)

    # tqdm진행률
    with tqdm(total=700) as pbar:
        for epoch in range(700):
            model.train(texts, total_examples=len(texts), epochs=1)
            model.alpha -= 0.001
            model.min_alpha = model.alpha
            pbar.update(1)
   
    model.save(model_path)
    np.save(vectors_ngrams_path, model.wv.vectors_ngrams, allow_pickle=False)

    print("학습 및 저장 완료")
    print("-" * 20)

#누적 학습용    
def update_fasttext_model(texts):
    print("-" * 20)
    print("모델 업데이트 시작")

    model = FastText.load(model_path)
    model.build_vocab(texts, update=True)
    model.train(texts, total_examples=model.corpus_count, epochs=350)

    with tqdm(total=500) as pbar:
        for epoch in range(500):
            model.train(texts, total_examples=model.corpus_count, epochs=1)
            model.alpha -= 0.001
            model.min_alpha = model.alpha
            pbar.update(1)

    model.save(model_path)
    np.save(vectors_ngrams_path, model.wv.vectors_ngrams, allow_pickle=False)
    print("모델 업데이트가 완료되었습니다.")
    print("-" * 20)

def main():
    freeze_support()

    with open(data_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    print(f"읽은 행 수: {len(data)}")

    ReToken_texts = preprocess(data)
    train_fasttext_model(ReToken_texts)
    # if not os.path.exists(model_path):
    #     train_fasttext_model(ReToken_texts)
    # else:
    #     update_fasttext_model(ReToken_texts)
    
    model = KeyedVectors.load(model_path)
    model.wv.vectors_ngrams = np.load(vectors_ngrams_path, allow_pickle=False)

# ##스위치
# if __name__ == '__main__':
#     main()