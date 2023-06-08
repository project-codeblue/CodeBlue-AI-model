from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Okt()
text = '''예전에도 오른쪽 머리 부분의 통증 병력(+)
1DA 오른쪽 머리 뒷 부분의 통증 발생,
콕콕 쑤시는 양상.
최근에 감기에 걸렸으며, 약간의 오심(+).
약을 먹지 는 않음.
신경을 약간 쓰는 편임.'''

# 형태소 분석
tokens = tokenizer.morphs(text)
print(tokens)

# 불용 제거
stopwords = set(['부분', '약간', '편임', '의'])

# 명사만 남긴 것에 벡터화
filtered_tokens = [ token for token in tokens if token not in stopwords and tokenizer.pos(token)[0][1] == 'Noun' ]
print(filtered_tokens)

tokenizer_keras = Tokenizer()
tokenizer_keras.fit_on_texts([filtered_tokens])

encoded = tokenizer_keras.texts_to_sequences([filtered_tokens])[0]
print(encoded)

# 패딩
max_length = 30
padded = pad_sequences([encoded], maxlen=max_length)
print(padded)



# 유사단어 훈련방식?!