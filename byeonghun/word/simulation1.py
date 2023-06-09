##현재 학습 데이터 자체가 각혈에 관한 유사데이터만 추출할 수 있도록 데이터로 학습되게끔 만듬

# import matplotlib.pyplot as plt #그래프 도출

# from hanspell import spell_checker #맞춤법 검사용(pip install git+https://github.com/jungin500/py-hanspell)
from tqdm.notebook import tqdm #pip install tqdm
from konlpy.tag import Okt
from konlpy.tag import Kkma
from gensim.models.word2vec import Word2Vec #학습모델
import re
import os

okt = Okt() # 빨라 써
# kkma = Kkma() #느려 안써

# 현재 스크립트 파일의 경로를 기준으로 상대 경로 계산
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'data.txt')
#data가져오기
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.readlines()

# 불용어 정의
stopwords = ['의','씩','증도','닮다','다','없다','고','엔','힘들다','액체','까지','두','야하다','오늘','아침','오늘아침','커지다','급격하다','치가','나타나다','조','이상승','이송이','병원','력','지속','함피','함','즉시','집중','떨어지다','하고','심하다','인','적','며','필요하다','되어다','되었다','인하다','심해지다','계속','있다','심각하다','점점','때문','을','로','상','하','지다','않다','이다','되다','에서','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','아침','또','또한','가끔','감','코',"계,'속",'현재','이어진','상황']
ReToken_texts =[]
clean_texts = []

# 전처리 1 : 단어 이외의 것 삭제
for line in data:
    line = re.compile('[^가-힣]').sub('', line)
    if line:
        clean_texts.append(line)

# 전처리 2 : 단어 바르게 변환, None값 제외, 불용어 제거
for text in tqdm(clean_texts,desc="Preprocessing"):
# for text in clean_texts :
    # spelled_text = spell_checker.check(text)
    tokenized_sentence = okt.morphs(text, stem=True)
    stopwords_removed_texts = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    ReToken_texts.append(stopwords_removed_texts)
# print("========>",ReToken_texts)

#모델 학습 훈련
model = Word2Vec(vector_size=100, window=5, min_count=4, workers=4, sg=1)
 # sg=0은 CBOW(Continuous Bag-of-Words) 알고리즘, sg=1은 Skip-gram 알고리즘 사용함
model.build_vocab(ReToken_texts)
model.train(ReToken_texts, total_examples=model.corpus_count, epochs=20000)#20000번 훈련

#해당 단어와 유사한 값 확인
similar_words = model.wv.most_similar("각혈") #현재 학습 데이터 자체가 각혈에 관한 유사데이터만 추출할 수 있도록 데이터로 학습되게끔 만듬
for word, similarity in similar_words:
    print(word, similarity)