# vscode에서는 제대로 실행이 안됨, 코랩에서 실행 가능
# 이렇게 오차율 계산을 하는 이유는 적절한 임계값(threshold)을 찾기 위해서 진행
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

input_img = Input(shape=(300, 300, 3))

# 인코더 부분
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 디코더 부분
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder_temp = Model(input_img, decoded)

# 모델의 가중치 로드하기
autoencoder_temp.load_weights('autoencoder_cnn_model1234/variables/variables')


# 테스트용 이미지를 불러오고 전처리
error_img = tf.keras.preprocessing.image.load_img('image1cut.jpg', target_size=(300, 300))
error_img_array = tf.keras.preprocessing.image.img_to_array(error_img) / 255.0
error_img_array = np.expand_dims(error_img_array, axis=0)

# 이미지를 오토인코더에 통과시키기
reconstructed_img_array = autoencoder_temp.predict(error_img_array)

# 원본 이미지와 재구성된 이미지 사이의 재구성 오차 계산
reconstruction_error = np.mean(np.square(error_img_array - reconstructed_img_array))

reconstruction_error