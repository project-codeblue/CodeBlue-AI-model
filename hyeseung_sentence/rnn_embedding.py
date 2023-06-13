from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense ,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import random
from real_data import data

# TensorBoard 로그 저장 디렉토리 설정
log_dir = "logs"

# TensorBoard 콜백 정의
tensorboard_callback = TensorBoard(log_dir=log_dir)

# 데이터 섞기
random.shuffle(data)

# 데이터 분리
symptoms, labels = zip(*data)

# 텍스트 데이터 전처리
tokenizer = Tokenizer()
tokenizer.fit_on_texts(symptoms)
encoded_symptoms = tokenizer.texts_to_sequences(symptoms)
word_index = tokenizer.word_index
num_words = len(word_index) + 1

# 패딩
max_length = max(len(seq) for seq in encoded_symptoms)
padded_symptoms = pad_sequences(encoded_symptoms, maxlen=max_length, padding='post')

# 응급 정도 레이블 전처리
num_classes = 5
encoded_labels = to_categorical(np.array(labels) - 1, num_classes=num_classes)

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(padded_symptoms, encoded_labels, test_size=0.2, random_state=42)

# RNN 모델 구성 (100차원, 활성화 함수:softmax)
embedding_dim = 100
model = Sequential()
model.add(Embedding(num_words, embedding_dim, input_length=max_length))
model.add(LSTM(64)) # hidden layer: 조정 대상
# model.add(LSTM(64, kernel_regularizer=regularizers.l1(0.01))) #L1 규제 적용.... 더 떨어짐ㅜ
model.add(Dropout(0.5)) # dropout - 과적합 방지: 조정 대상
model.add(Dense(num_classes, activation='softmax'))

# 모델 컴파일 (알고리즘:adam, 손실함수:categorical_crossentropy, 평가지표:accuracy)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습률 스케줄링 함수 정의 (100번동안은 학습률 유지 후 0.1씩 감소 -> 초기학습은 빠르게)
def lr_scheduler(epoch, lr):
    if epoch < 1000:
        return lr
    else:
        return lr * 0.1 # learning rate: 조정 대상

# 학습률 스케줄링 콜백 정의
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

class CustomEarlyStopping(Callback):
    def __init__(self, accuracy_threshold=0.95, patience=30):
        super(CustomEarlyStopping, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.patience = patience
        self.wait = 0 # 개선 없는 횟수 세기
        self.stopped_epoch = 0 # 종료 에포크 번호
        self.best_weights = None # 최적 가중치 저장

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')

        if current_accuracy >= self.accuracy_threshold and self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print(f"\n조기 종료: 정확도 {self.accuracy_threshold} 이상에 도달하고 {self.patience}번 동안 개선되지 않았습니다.")
            print(f"{self.patience}번 이전의 모델 가중치로 복원합니다.")
            self.model.set_weights(self.best_weights)

        if current_accuracy is not None:
            if self.best_weights is None or current_accuracy > self.best_accuracy:
                self.best_weights = self.model.get_weights()
                self.best_accuracy = current_accuracy
                self.wait = 0
            else:
                self.wait += 1

# 조기 종료 콜백 정의 (10번통안 검증손실이 개선되지 않으면 조기종료)
custom_early_stopping = CustomEarlyStopping(accuracy_threshold=0.95, patience=30)

# 학습 (반복횟수:1000, 한번에 처리할 데이터 샘플:32)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), # epochs: 조정 대상
          callbacks=[custom_early_stopping, lr_scheduler_callback, tensorboard_callback], verbose=1)

# 성능 평가
loss, accuracy = model.evaluate(X_test, y_test)
print("테스트 손실:", loss)
print("테스트 정확도:", accuracy)

# 예측
sample_symptom = ["오른쪽 귀 뒷부분에 몽우리 손가락 한마디 정도가 만져지고 점점 커짐. 만지면 아픔. 볼부위가 많이 부었다. 발열 기침 가래 피부병변. 최근에 감기 걸린 증상이 없었다"]
encoded_sample = tokenizer.texts_to_sequences([sample_symptom])
padded_sample = pad_sequences(encoded_sample, maxlen=max_length, padding='post')
prediction = model.predict(padded_sample)
emergency_level = np.argmax(prediction, axis=1) + 1
print("환자의 응급 정도:", emergency_level)
