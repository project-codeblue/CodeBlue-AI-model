from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from gensim.models import Word2Vec
import numpy as np

# 훈련 데이터
texts = [
    "밥을 먹다가 아랫입술이 경련이 난 것처럼 떨린다.",
    "10분 정도 한 후에 괜찮아짐.",
    "월요일 체력단련 후 명치가 아프면서 밤새 동안 구토, 구역질",
    "교정하고 있는 상태."
]

# 응급 정도 레이블
labels = [4, 2, 3, 1]

# 문장 토큰화
okt = Okt()
sentences = [okt.morphs(text) for text in texts]

# 불용 제거
stopwords = ['부분', '약간', '편임', '의']
filtered_sentences = [[word for word in sentence if word not in stopwords] for sentence in sentences]

# Word2Vec 모델 학습 (100차원, 주변 단어고려:3, 최소단어빈도:1, 스레드:4, skip-gram(1)알고리즘 사용. CBOW(0)으로 변경 가능)
embedding_model = Word2Vec(filtered_sentences, vector_size=100, window=3, min_count=1, workers=4, sg=1)

# 벡터화된 문장 획득
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_sentences)
encoded_sentences = tokenizer.texts_to_sequences(filtered_sentences)
padded_sentences = pad_sequences(encoded_sentences, padding='post')

# 응급 정도 레이블을 배열로 변환
labels = np.array(labels)

# 모델 구성
model = Sequential()
model.add(Flatten(input_shape=(padded_sentences.shape[1],))) 
model.add(Dense(64, activation='relu')) # 입력노드 64, 활성화 함수 ReLU
model.add(Dense(5, activation='softmax')) # 출력노드 5(응급도가 5단계니까), 활성화 함수 softmax 

# 모델 컴파일 (알고리즘:adam, 손실함수:categorical_crossentropy, 평가지표:accuracy)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 (반복횟수:10, 한번에 처리할 데이터 샘플:32)
model.fit(padded_sentences, labels, epochs=10, batch_size=32)

# 환자 상태 입력
new_text = "밤에 가슴이 답답하고 호흡이 어려움"

# 입력 문장 전처리
new_sentence = okt.morphs(new_text)
filtered_new_sentence = [word for word in new_sentence if word not in stopwords]
encoded_new_sentence = tokenizer.texts_to_sequences([filtered_new_sentence])
padded_new_sentence = pad_sequences(encoded_new_sentence, maxlen=padded_sentences.shape[1], padding='post')

# 응급 정도 예측
prediction = model.predict(padded_new_sentence)
emergency_level = np.argmax(prediction) + 1
print("환자의 응급 정도:", emergency_level)
