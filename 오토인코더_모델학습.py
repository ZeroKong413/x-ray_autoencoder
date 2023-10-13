import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.callbacks import EarlyStopping

# CNN 기반의 오토인코더 모델 정의
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

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# 학습을 할때 10번 이상 loss 값이 안바뀌면 중간에 학습을 일찍 중지시킴
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

# 모델 학습
autoencoder.fit(all_images_array, all_images_array, epochs=50, batch_size=10, callbacks=[early_stopping])

# 학습된 모델을 저장
# autoencoder.save('autoencoder_cnn_model3') # 이게 저장할때 경로에 한글이 있어서 저장이 제대로 안됨
save_path = "C:\\Users\\asdfa\\OneDrive\\2023\\autoencoder_cnn_model1234"
# autoencoder.save(save_path)
save_model(autoencoder, save_path)