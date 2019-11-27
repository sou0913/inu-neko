# 画像生成

import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def draw_images(generator, x, dir_name, index):
    save_name = 'extended-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=save_name, save_format='jpg')

    for i in range(5):
        bach = g.next()

if __name__ == '__main__':

    output_dir = "extended"
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    images = glob.glob(os.path.join("./","*.jpg"))

    generator = ImageDataGenerator(
                rotation_range=45,
                width_shift_range=0.3,
                height_shift_range=0.3,
                channel_shift_range=50.0,
                shear_range=0.39,
                horizontal_flip=True)

    for i in range(len(images)):
        img = load_img(images[i])
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        draw_images(generator, x, output_dir, i)

