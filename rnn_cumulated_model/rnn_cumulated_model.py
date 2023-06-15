from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import random, urllib.request, pandas as pd, pickle, re
from konlpy.tag import Okt
import matplotlib.pyplot as plt
from dataset_second import data

# TensorBoard 
### TensorBoard 로그 저장 디렉토리 설정
log_dir = "logs"

### TensorBoard 콜백 정의
tensorboard_callback = TensorBoard(log_dir=log_dir)


# 데이터 전처리 
### 데이터 섞기
random.shuffle(data)

### duplicated data 제거
seen_values = set()
data = [item for item in data if item[0] not in seen_values and not seen_values.add(item[0])]

### 데이터 분리
symptoms_before_tuning, labels = zip(*data)
print("TOTAL_DATASET: ", len(symptoms_before_tuning))


# 토큰화
stopwords = [',','.','의','로','을','가','이','은','들','는','성','좀','잘','걍','과','고','도','되','되어','되다','를','으로','자','에','와','한','합니다','입니다','있습니다','니다','하다','임','음','환자','응급','응급실','이송','상황','상태','증상','증세','구조']
okt = Okt()

### stopword 제거, 토큰화
symptoms = []
for sentence in symptoms_before_tuning:
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    symptoms.append(stopwords_removed_sentence)


# 토크나이저 불러오기
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

encoded_symptoms = tokenizer.texts_to_sequences(symptoms)


# 패딩
MAX_LEN = 16
max_length = max(len(seq) for seq in encoded_symptoms)
print(max_length)
padded_symptoms = pad_sequences(encoded_symptoms, maxlen=MAX_LEN, padding='post', truncating='post')


# 응급 정도 레이블 전처리
num_classes = 5
encoded_labels = to_categorical(np.array(labels) - 1, num_classes=num_classes)


# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(padded_symptoms, encoded_labels, test_size=0.2, random_state=42)


# 기존 모델 불러오기
model = load_model('rnn_model.h5')


# 학습률 스케줄링 함수 정의 (100번동안은 학습률 유지 후 0.1씩 감소 -> 초기학습은 빠르게)
def lr_scheduler(epoch, lr):
    if epoch < 1000:
        return lr
    else:
        return lr * 0.1 # learning rate: 조정 대상

### 학습률 스케줄링 콜백 정의
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)


# 조기 종료 콜백 정의
class CustomEarlyStopping(Callback):
    def __init__(self, accuracy_threshold=0.95, patience=30):
        super(CustomEarlyStopping, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.patience = patience
        self.wait = 0  # 개선 없는 횟수 세기
        self.stopped_epoch = 0  # 종료 에포크 번호
        self.best_weights = None  # 최적 가중치 저장
        self.best_val_loss = float('inf')  # 최적의 검증 손실 초기화

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')
        current_val_loss = logs.get('val_loss')

        if current_accuracy >= self.accuracy_threshold and self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print(f"\n조기 종료: 정확도 {self.accuracy_threshold} 이상에 도달하고 {self.patience}번 동안 검증 손실이 개선되지 않았습니다.")
            print(f"{self.patience}번 이전의 모델 가중치로 복원합니다.")
            self.model.set_weights(self.best_weights)

        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_weights = self.model.get_weights()
                self.best_val_loss = current_val_loss
                self.wait = 0
            else:
                self.wait += 1

### 조기 종료 콜백 사용 (30번 동안 검증손실이 개선되지 않으면 조기종료)
early_stopping_callback = CustomEarlyStopping(accuracy_threshold=0.98, patience=30)


# 모델 체크포인트 - ModelCheckpoint를 사용하여 검증 데이터의 정확도(val_accuracy)가 이전보다 좋아질 경우에만 모델을 저장
# mc = ModelCheckpoint('rnn_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


# 모델 컴파일 (알고리즘:adam, 손실함수:categorical_crossentropy, 평가지표:accuracy)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 학습 (반복횟수:1000, 한번에 처리할 데이터 샘플:32)
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), # epochs: 조정 대상
          callbacks=[early_stopping_callback, lr_scheduler_callback, tensorboard_callback], verbose=1) 


# 성능 평가
loss, accuracy = model.evaluate(X_test, y_test)
print("테스트 손실:", loss)
print("테스트 정확도:", accuracy)


# 문장 예측
def emergency_level_prediction(sample_sentence):
    # 샘플 문장 전처리
    sample_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', sample_sentence)
    sample_sentence = okt.morphs(sample_sentence, stem=True) # 토큰화
    sample_sentence = [word for word in sample_sentence if not word in stopwords] # 불용어 제거
    # 샘플 문장을 토큰화하고 패딩
    encoded_sample = tokenizer.texts_to_sequences([sample_sentence])
    padded_sample = pad_sequences(encoded_sample, maxlen=MAX_LEN, padding='post')
    # 샘플 문장 응급도 예상
    prediction = model.predict(padded_sample)
    emergency_level = np.argmax(prediction, axis=1) + 1
    confidence = prediction[0][emergency_level[0]-1] # 각 클래스의 확률 중에서 선택된 클래스의 확률
    print(f"응급도: {emergency_level[0]}, 확신도: {confidence * 100.0}%")


# 예시 문장
emergency_level_prediction("응급환자는 심장마비로 인해 의식을 잃고 쓰러졌습니다. 호흡 곤란 상태입니다.") # 1
emergency_level_prediction("환자는 현재 쇼크로 인한 무의식 상태입니다. 바로 응급실로 이동해야하는 위급상황입니다.")
emergency_level_prediction("지금 환자의 혈액 순환이 장애가 생겼습니다. 환자는 혈류가 약해져 무기력한 상태입니다.") # 2
emergency_level_prediction("환자는 뇌출혈로 인한 뇌졸중으로 판단됌. 응급실로 이동중.") 
emergency_level_prediction("환자의 맥박수가 매우 높은것으로 판단됌. 정상적인 맥박이 아님") # 3
emergency_level_prediction("환자는 흑색변과 탈수 증세를 보임") 
emergency_level_prediction("배뇨 장애를 가진 환자가 탑승. 요로감염으로 의심됌") # 4
emergency_level_prediction("유해물질을 먹은 것 같은데 큰 증상을 보이지 않지만 응급실로 이동중") 
emergency_level_prediction("설사로 인한 복통과 탈수 증상") # 5
emergency_level_prediction("부종으로 인해 움직임의 어려움을 느끼고 있는 환자가 탑승 중입니다.")

# 모델 저장
model.save('rnn_cumulated_model.h5')
