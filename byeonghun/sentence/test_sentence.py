from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# text = '''밥을 먹다가 아랫입술이 경련이 난 것처럼 떨린다. 
# 10분 정도 한 후에 괜찮아짐. 
# 월요일 체력단련 후 명치가 아프면서 밤새 동안 구토, 구역질 
# 교정하고 있는 상태.'''

text = '''예전에도 오른쪽 머리 부분의 통증 병력(+)
1DA 오른쪽 머리 뒷 부분의 통증 발생,
콕콕 쑤시는 양상.
최근에 감기에 걸렸으며, 약간의 오심(+).
약을 먹지 는 않.
신경을 약간 쓰는 편임.'''

# 문장 토큰화
okt = Okt()
sentences = text.split("\n")

# 형태소 분석
morph_sentences = [okt.morphs(sentence) for sentence in sentences]
print(morph_sentences)

# 명사 추출
nouns_sentences = [okt.nouns(sentence) for sentence in sentences]
print(nouns_sentences)

# 불용 제거
stopwords = ['부분', '약간', '편임', '의']
filtered_sentences = [[word for word in sentence if word not in stopwords] for sentence in nouns_sentences]
print(filtered_sentences)

# 출현 빈도수 확인
word_counts = {}
for sentence in filtered_sentences:
    for word in sentence:
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1

# 출현 빈도수가 높은 상위 n개 단어 추출
top_n = 3
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
top_words = [word for word, count in sorted_word_counts[:top_n]]
print(top_words)

# 명사만 남긴 것에 벡터화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_sentences)
encoded_sentences = tokenizer.texts_to_sequences(filtered_sentences)
print(encoded_sentences)

# 패딩
MAX_LENGTH = max(len(s) for s in encoded_sentences)
padded_sentences = pad_sequences(encoded_sentences, maxlen=MAX_LENGTH, padding='post')
print("Padded sentences:\n", padded_sentences)

# Word2Vec 모델 학습
embedding_model = Word2Vec(filtered_sentences, vector_size=100, window=3, min_count=1, workers=4, sg=1)
# 학습된 임베딩 벡터 확인
word_vectors = embedding_model.wv
print("단어 벡터:")
for word, idx in word_vectors.key_to_index.items():
    print(word)
    # print(word, word_vectors.vectors[idx])