from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import random, urllib.request, pandas as pd, pickle, re
from konlpy.tag import Okt
okt = Okt()

# 전처리용 상수
MAX_LEN = 16
stopwords = [',','.','의','로','을','가','이','은','들','는','성','좀','잘','걍','과','고','도','되','되어','되다','를','으로','자','에','와','한','합니다','입니다','있습니다','니다','하다','임','음','환자','응급','상황','상태','증상','증세','구조']

# 기존 모델 불러오기
model = load_model('rnn_codeblue_model.h5')

# 토크나이저 불러오기
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

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
emergency_level_prediction("의식이 없으며, 심부전증도 심하고, 뇌경색 증상도 보이며, 사지마비까지 있음, 계속되는 발작으로 혼란상태 유지됨, 심지어 청각손실과 가슴통증까지 있음") # 1
emergency_level_prediction("지금 환자의 혈액 순환이 장애가 생겼습니다. 환자는 혈류가 약해져 무기력한 상태입니다.") # 2
emergency_level_prediction("환자의 맥박수가 매우 높은것으로 판단됌. 정상적인 맥박이 아님") # 3
emergency_level_prediction("배뇨 장애를 가진 환자가 탑승. 요로감염으로 의심됌") # 4
emergency_level_prediction("설사로 인한 복통과 탈수 증상") # 5
