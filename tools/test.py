# モデルアウトプットテスト
import tensorflow as tf
keras = tf.keras
import numpy as np
# 画像の整形。RGBは０〜255オーダー、MovileNetV2は-1〜1で計算する。
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3) 
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    image *= 2.0
    image -= 1.0
    return image
 
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)
# 訓練済みのモデルでテスト画像を解析し、返された確率を0〜100%の形に整形して出力。
model = keras.models.load_model('inu.h5')
image = load_and_preprocess_image("/Users/so/Downloads/70fe389b16b87b0bd80ac5ec84faf7b9.jpg")
image = (np.expand_dims(image,0))
preds = model.predict(image)
print(round(preds[0][1]*100))
