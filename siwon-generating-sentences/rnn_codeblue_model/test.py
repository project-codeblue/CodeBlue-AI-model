from konlpy.tag import Okt

# stopwords 정의
stopwords = ['은', '는', '이', '가', '을', '를', '환자', '합니다', '니다', '하다']

# 문장 전처리 함수
def preprocess_sentence(sentence):
    # 형태소 분석기로 문장을 토큰화
    okt = Okt()
    tokens = okt.morphs(sentence, stem=True)
    
    # stopwords 제거
    tokens = [token for token in tokens if token not in stopwords]
    
    # 전처리된 문장 반환
    preprocessed_sentence = ' '.join(tokens)
    return preprocessed_sentence

# 예시 문장 전처리
sentence = '환자는 심각한 가슴 통증을 호소합니다.'
preprocessed = preprocess_sentence(sentence)
print(preprocessed)
