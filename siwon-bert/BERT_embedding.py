import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from data import data

# 우리가 사용할 모델
MODEL_NAME = "bert-base-multilingual-cased"

# 데이터 분리
symptoms = [i[0] for i in data]
labels = [i[1] for i in data]

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(symptoms, labels, test_size=0.2, random_state=42, stratify=labels)

# BERT 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 입력 데이터를 BERT 입력 형식으로 변환하는 함수
# 토큰 인코딩, 패딩 등 작업 수행됨
# 토큰 출력해보고싶당
def convert_examples_to_features(texts, labels, tokenizer):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'][0]) 
        attention_masks.append(encoded['attention_mask'][0])

        print("텍스트:", text)
        print("인코딩된 토큰:", tokenizer.convert_ids_to_tokens(encoded['input_ids'][0]))  

    return tf.constant(input_ids), tf.constant(attention_masks), tf.constant(labels)  

# 데이터를 BERT 입력 형식으로 변환
train_input_ids, train_attention_masks, train_labels = convert_examples_to_features(X_train, y_train, tokenizer)
test_input_ids, test_attention_masks, test_labels = convert_examples_to_features(X_test, y_test, tokenizer)

# BERT 모델 로드
bert_model = TFAutoModel.from_pretrained(MODEL_NAME)

# BERT 모델 구성
input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32)
attention_masks = tf.keras.Input(shape=(128,), dtype=tf.int32)
outputs = bert_model(input_ids, attention_mask=attention_masks)[0]
outputs = tf.keras.layers.GlobalMaxPool1D()(outputs)
outputs = tf.keras.layers.Dense(5, activation='softmax')(outputs)

# 전체 모델 정의
model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=outputs)

# 모델 컴파일 (알고리즘:adam, 손실함수:sparse_categorical_crossentrop, 평가지표:accuracy)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습 (반복횟수:5, 배치크기:32)
model.fit(
    [train_input_ids, train_attention_masks],
    train_labels,
    validation_data=([test_input_ids, test_attention_masks], test_labels),
    epochs=5,
    batch_size=32
)

# 성능 평가
loss, accuracy = model.evaluate([test_input_ids, test_attention_masks], test_labels)
print("테스트 손실:", loss)
print("테스트 정확도:", accuracy)

# 예측
sample_symptom = ["오른쪽 머리 뒷 부분의 통증 발생"]
sample_input_ids, sample_attention_masks, _ = convert_examples_to_features(sample_symptom, [0], tokenizer)
prediction = model.predict([sample_input_ids, sample_attention_masks])
emergency_level = tf.argmax(prediction, axis=1).numpy()[0] + 1
print("환자의 응급 정도:", emergency_level)


# 학습리셋하는 방법
# TensorFlow의 경우 tf.keras.Model의 initialize_weights 메서드를 사용하여 가중치를 초기화
# tf.keras.optimizers.Optimizer의 get_weights 메서드를 사용하여 옵티마이저의 상태를 가져올 수 있고, 이를 set_weights 메서드를 사용하여 초기화
# 학습 반복 횟수 초기화: 이전에 학습한 모델의 학습 반복 횟수를 초기화합니다. 즉, 에포크(epoch) 수를 0으로 설정하여 처음부터 학습을 시작