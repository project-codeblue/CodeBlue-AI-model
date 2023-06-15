# -- coding: utf-8 --
# ↑상단의 주석 지우지 마세요. python이 읽는 주석입니다.

from konlpy.tag import Okt
from multiprocessing import freeze_support
from gensim.models.fasttext import FastText
import re
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from stop_word import stop_words,compound_words

okt = Okt()

#절대 경로 정리
model_path = "yongjae_word/model/fasttext.model"
vectors_ngrams_path = "yongjae_word/model/fasttext.model.wv.vectors_ngrams.npy"
data_path = 'C:/AI/CodeBlue-AI/yongjae_word/data.txt'

def preprocess(data):
    ReToken_texts = []
    for line in data:
        line = re.compile('[^가-힣]').sub(' ', line)
        token_text = okt.morphs(line)
        stopWords_removed_texts = [word for word in token_text if not word in stop_words]
        compoundWords_removed_texts = [word for word in stopWords_removed_texts if not word in compound_words]
        # print("전처리 완료==>",compoundWords_removed_texts)
        if compoundWords_removed_texts:
             ReToken_texts.append(compoundWords_removed_texts)
    return ReToken_texts

#모델 초기화 & 초기 학습용
def train_fasttext_model(texts):
    print("-" * 20)
    print("학습 시작")
    model = FastText(vector_size=100, window=5, min_count=6, workers=4, sg=1)

    model.build_vocab(texts)

    with tqdm(total=1500) as pbar:
        for epoch in range(1000):
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
    print("학습 업데이트 시작")

    model = FastText.load(model_path)
    model.window = 5
    model.min_count = 6
    model.workers = 4
    model.build_vocab(texts, update=True)
    model.train(texts, total_examples=model.corpus_count, epochs=1000)

    with tqdm(range(1500)) as pbar:
        for epoch in pbar:
            model.train(texts, total_examples=model.corpus_count, epochs=1)
            model.alpha -= 0.001
            model.min_alpha = model.alpha
            # pbar.update(1)

    model.save(model_path)
    np.save(vectors_ngrams_path, model.wv.vectors_ngrams, allow_pickle=False)
    model.wv.vectors_ngrams = np.load(vectors_ngrams_path, allow_pickle=False)

    print("모델 업데이트가 완료되었습니다.")
    print("-" * 20)


# ##그래프화
# def visualize_similarity(model):
#     word_list = list(model.wv.key_to_index)
#     random.shuffle(word_list)
#     selected_words = word_list[:200]

#     vectors = np.array([model.wv[word] for word in selected_words])
#     tsne = TSNE(n_components=2, random_state=0)
#     vectors_2d = tsne.fit_transform(vectors)

#     plt.figure(figsize=(12, 12))
#     for point, word in zip(vectors_2d, selected_words):
#         plt.scatter(point[0], point[1])
#         plt.annotate(word, xy=(point[0], point[1]))
#     plt.show()

def main():
    freeze_support()

    with open(data_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    print(f"읽은 행 수: {len(data)}")

    ReToken_texts = preprocess(data)
    # train_fasttext_model(ReToken_texts) ##새학습
    # if not os.path.exists(model_path):
    #     train_fasttext_model(ReToken_texts)
    # else:
    update_fasttext_model(ReToken_texts) ##누적학습

    # model = KeyedVectors.load(model_path) #삭제해도될지도?
    # model.wv.vectors_ngrams = np.load(vectors_ngrams_path, allow_pickle=False) #삭제해도될지도?
    # visualize_similarity(model)

#스위치
# if __name__ == '__main__':
#         main()