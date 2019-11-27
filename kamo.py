import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
PATH = "/Users/so/Desktop/projects/tf-app/kamo_materials"
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
train_karugamo_dir = os.path.join(train_dir, 'karugamo')
train_magamo_dir = os.path.join(train_dir, 'magamo')
train_magamo_e_dir = os.path.join(train_dir, "magamo_e")
train_magamo_mesu_dir = os.path.join(train_dir, "magamo_mesu")

validation_karugamo_dir = os.path.join(validation_dir, 'karugamo')
validation_magamo_dir = os.path.join(validation_dir, 'magamo')
validation_magamo_e_dir = os.path.join(validation_dir, "magamo_e")
validation_magamo_mesu_dir = os.path.join(validation_dir, "magamo_mesu")


num_karugamo_tr = len(os.listdir(train_karugamo_dir))
num_magamo_tr = len(os.listdir(train_magamo_dir))
num_magamo_e_tr = len(os.listdir(train_magamo_e_dir))
num_magamo_mesu_tr = len(os.listdir(train_magamo_mesu_dir))

num_karugamo_val = len(os.listdir(validation_karugamo_dir))
num_magamo_val = len(os.listdir(validation_magamo_dir))
num_magamo_e_val = len(os.listdir(validation_magamo_e_dir))
num_magamo_mesu_val = len(os.listdir(validation_magamo_mesu_dir))

total_train = num_karugamo_tr + num_magamo_tr + num_magamo_e_tr + num_magamo_mesu_tr
total_val = num_karugamo_val + num_magamo_val + num_magamo_e_val + num_magamo_mesu_val

batch_size = 32
epochs = 20
IMG_HEIGHT = 150
IMG_WIDTH = 150
image_gen = ImageDataGenerator(rescale=1./255,
                               horizontal_flip=True,
                               rotation_range=60,
                               width_shift_range=.30,
                               height_shift_range=.30,
                               zoom_range=0.8,
                               shear_range=35)
image_gen_val = ImageDataGenerator(rescale=1./255)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                               class_mode='sparse')
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='sparse')
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
new_model = keras.models.load_model('kamo_model.h5')

predict = new_model.predict(val_data_gen)
print(predict[0])
#history = model.fit_generator(
#    train_data_gen,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    validation_data=val_data_gen,
#    validation_steps=total_val // batch_size
#)
#model.save('kamo_model.h5')
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs_range = range(epochs)
#
#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(1, 2, 2)
#plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show()


#モデルを確認する
#model.summary()



#ジェネレーターで画像を処理できているか確認
#sample_training_images, _ = next(train_data_gen)
#def plotImages(images_arr):
#    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
#    axes = axes.flatten()
#    for img, ax in zip(images_arr, axes):
#        ax.imshow(img)
#        ax.axis('off')
#    plt.tight_layout()
#    plt.show()
#plotImages(sample_training_images[:5])


# 画像の数を表示

#print('total training karugamo images:', num_karugamo_tr)
#print('total training magamo images:', num_magamo_tr)
#print('total training magamo_e images:', num_magamo_e_tr)
#print('total training magamo_mesu images:', num_magamo_mesu_tr)
#print('total training kamonegi images:', num_kamonegi_tr)
#print('total validation karugamo images:', num_karugamo_val)
#print('total validation magamo images:', num_magamo_val)
#print('total validation magamo_e images:', num_magamo_e_val)
#print('total validation magamo_mesu images:', num_magamo_mesu_val)
#print('total validation kamonegi images:', num_kamonegi_val)



