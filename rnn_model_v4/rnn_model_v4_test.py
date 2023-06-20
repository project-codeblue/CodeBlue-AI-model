from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pickle, re
from konlpy.tag import Okt
import time
okt = Okt()

start_time = time.time()

# 전처리용 상수
MAX_LEN = 18
stopwords = [',','.','의','로','을','가','이','은','들','는','성','좀','잘','걍','과','고','도','되','되어','되다','를','으로','자','에','와','한','합니다','입니다','있습니다','니다','하다','임','음','환자','응급','상황','상태','증상','증세','구조']

# 기존 모델 불러오기
model = load_model('rnn_model_v4_no_cum.h5')

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
# emergency_level_prediction("응급환자는 심장마비로 인해 의식을 잃고 쓰러졌습니다. 호흡 곤란 상태입니다.") # 예상: 1
# emergency_level_prediction("환자는 현재 쇼크로 인한 무의식 상태입니다. 바로 응급실로 이동해야하는 위급상황입니다.") # 예상: 1
# emergency_level_prediction("지금 환자의 혈액 순환이 장애가 생겼습니다. 환자는 혈류가 약해져 무기력한 상태입니다.") # 예상: 2
# emergency_level_prediction("환자는 뇌출혈로 인한 뇌졸중으로 판단됌. 응급실로 이동중.") # 예상: 2
# emergency_level_prediction("환자의 맥박수가 매우 높은것으로 판단됌. 정상적인 맥박이 아님") # 예상: 3
# emergency_level_prediction("환자는 흑색변과 탈수 증세를 보임") # 예상: 3
# emergency_level_prediction("배뇨 장애를 가진 환자가 탑승. 요로감염으로 의심됌") # 예상: 4
# emergency_level_prediction("유해물질을 먹은 것 같은데 큰 증상을 보이지 않지만 응급실로 이동중") # 예상: 4
# emergency_level_prediction("설사로 인한 복통과 탈수 증상") # 예상: 5
# emergency_level_prediction("부종으로 인해 움직임의 어려움을 느끼고 있는 환자가 탑승 중입니다.") # 예상: 5

# emergency_level_prediction("환자가 생리통을 호소하고 있습니다. 특히 오한을 느끼고 있습니다.") # 예상: 4
# emergency_level_prediction("응급 환자는 질통증 증상을 보임. 소변을 볼 때 고통이 심해짐") # 예상: 5
# emergency_level_prediction("임산부 환자. 20주 이상. 자궁경부 압박으로 인한 고통") # 예상: 2
# emergency_level_prediction("분만 직전의 환자. 특이사항은 없음") # 예상: 2
# emergency_level_prediction("불면증 환자로 자기 전 걱정과 과한 생각으로 인해 과도하게 피곤해함") # 예상: 4

emergency_level_prediction("머리가 어지럽고 혼미한 상태입니다. 가슴이 아프고 호흡이 어려워하며, 머리가 어지럽고 혼미해합니다.")
emergency_level_prediction("몸에 열이 나고 기침과 가래가 있으며, 흉통과 호흡이 어려워합니다.")
emergency_level_prediction("갑작스런 복통과 복부의 팽만감을 느끼며 토혈이 나오고, 혈압이 떨어지고 있습니다.")
emergency_level_prediction("갑작스럽게 어지러워지며 실신하고 구토를 하며, 두통을 겪고 있습니다.")
emergency_level_prediction("혼동과 경련을 겪으며 오심과 구토를 하며, 호흡이 빨라지고 있습니다.")
emergency_level_prediction("가슴이 아프고 호흡이 어려우며 맥박이 빨라지고, 피부가 청색으로 변하고 있습니다.")
emergency_level_prediction("목이 아프고 붓고 열이 나며, 목의 감각이 둔해지고 인후통을 느끼고 있습니다.")
emergency_level_prediction("혼란해하며 어지러움을 느끼며 구토하며, 피부에 발진이 나타나고 있습니다.")
emergency_level_prediction("몹시 심한 복통을 겪으며 설사를 하고, 안구가 충혈되고 통증을 느끼고 있습니다.")
emergency_level_prediction("체온이 낮아지고 혼수 상태에 빠지며, 호흡이 느려지고 심장 박동이 감소하고 있습니다.")
emergency_level_prediction("어깨가 아프고 호흡이 어려워하며 감각을 잃고, 손가락이 변색되고 있습니다.")
emergency_level_prediction("가슴에 압박감을 느끼며 흉통을 겪고 발한과 부종이 생기며 호흡이 빨라집니다.")
emergency_level_prediction("복부가 아프고 구토를 하며 혈뇨가 나오며 복부가 팽창되고 있습니다.")
emergency_level_prediction("맥박이 빨라지며 혼란해하며 호흡이 어려워하고 구토를 합니다.")
emergency_level_prediction("열이 나며 피부에 발진이 나오며 인후통을 느끼고 체중이 감소하고 있습니다.")
emergency_level_prediction("소화가 잘 안되고 복부가 아프며 체중이 감소하고 변비를 겪으며 위의 점막이 손상되었습니다.")
emergency_level_prediction("어깨에 급격한 통증이 있으며 기도가 수축되어 호흡이 어렵고 가려움과 피부 발진이 나타나고 있습니다.")
emergency_level_prediction("충수가 염증이 있으며 급격한 복통을 호소하며, 어제부터 설사와 혈뇨가 있고 현재 저혈압 상태입니다.")

end_time = time.time()
execution_time = end_time - start_time

print(f"실행 시간: {execution_time}초") # 4.4s
