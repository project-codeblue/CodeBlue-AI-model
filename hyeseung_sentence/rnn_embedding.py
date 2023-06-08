from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# 증상 데이터
symptoms = [
    "밥을 먹다가 아랫입술이 경련이 난 것처럼 떨린다.",
    "10분 정도 한 후에 괜찮아짐.",
    "월요일 체력단련 후 명치가 아프면서 밤새 동안 구토, 구역질",
    "교정하고 있는 상태."
]

# 응급 정도 레이블
labels = [4, 2, 3, 1]  
num_classes = 5
encoded_labels = to_categorical(labels, num_classes=num_classes)

# 텍스트 데이터 전처리
tokenizer = Tokenizer()
tokenizer.fit_on_texts(symptoms)
encoded_symptoms = tokenizer.texts_to_sequences(symptoms)
word_index = tokenizer.word_index
num_words = len(word_index) + 1

# 패딩
max_length = max(len(seq) for seq in encoded_symptoms)
padded_symptoms = pad_sequences(encoded_symptoms, maxlen=max_length, padding='post')
print(padded_symptoms)

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(padded_symptoms, encoded_labels, test_size=0.2, random_state=42)

# RNN 모델 구성 (100차원, 활성화 함수:softmax)
embedding_dim = 100
model = Sequential()
model.add(Embedding(num_words, embedding_dim, input_length=max_length))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='softmax'))

# 모델 컴파일 (알고리즘:adam, 손실함수:categorical_crossentropy, 평가지표:accuracy)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습 (반복횟수:10, 한번에 처리할 데이터 샘플:32)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 성능 평가
loss, accuracy = model.evaluate(X_test, y_test)
print("테스트 손실:", loss)
print("테스트 정확도:", accuracy)

# 예측
sample_symptom = ["오른쪽 머리 뒷 부분의 통증 발생"]
encoded_sample = tokenizer.texts_to_sequences([sample_symptom])
padded_sample = pad_sequences(encoded_sample, maxlen=max_length, padding='post')
prediction = model.predict(padded_sample)
emergency_level = np.argmax(prediction, axis=1) + 1
print("환자의 응급 정도:", emergency_level)


# 훈련데이터를 어떻게 모을것인가?

# 응급도를 결정할 때 마다 훈련을 시켜야하는데 미리 모델을 저장할수는 없는 것인가? 
# -> 로그인의 방식을 사용?
