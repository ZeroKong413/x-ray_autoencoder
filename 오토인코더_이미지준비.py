import numpy as np
import tensorflow as tf

# 훈련용 이미지 준비
image_paths = ['image001.jpg', 'image002.jpg', 'image003.jpg', 'image004.jpg', 'image005.jpg', 'image006.jpg', 'image007.jpg', 'image008.jpg', 'image009.jpg', 'image010.jpg']
all_images = []

# 이미지를 각각 10번 복사해서 훈련에 사용
for path in image_paths:
    img = tf.keras.preprocessing.image.load_img(path, target_size=(300, 300))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    for _ in range(10):
        all_images.append(img_array)

# 이미지 픽셀 값(0~255)을 0~1 사이로 정규화
all_images_array = np.array(all_images) / 255.0