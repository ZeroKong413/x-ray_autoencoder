import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 저장된 오토인코더 모델 불러오기
loaded_autoencoder = load_model('autoencoder_cnn_model1234')


# 테스트에 사용할 이미지를 로드하고 전처리
img_path = 'imagecut.jpg' # DB에서 이미지 파일을 끌고와야할 곳
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(300, 300))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 이미지 재구성
reconstructed = loaded_autoencoder.predict(img_array)

# 재구성 오차를 계산
mse = np.mean(np.square(img_array - reconstructed))

# 이상 탐지
threshold = 0.00021  # 예시임계값, 너무 낮추면 정상을 못찾아냄, 너무 높이면 다 정상이라고 함
if mse > threshold:  # 이거를 찾기 위해 아래와 같이 재구성 오차율을 계산
    print("이상 데이터입니다.")
else:
    print("정상 데이터입니다.")


