import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import os
import zipfile
# https://codetorial.net/tensorflow/classifying_the_cats_and_dogs.html


# 이미지 받고 동일 위치에 zip파일 압축해제
# 자료 다운로드 한번 다운하면 이곳은 주석  
# 하려고했으나 모델이 겹쳐 학습이 누적될까봐 매번 공장 초기화 하고 다운 받기로 하자

!wget --no-check-certificate \
https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
-O /tmp/cats_and_dogs_filtered.zip
local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()





# 기본 경로 설정
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


# 훈련에 사용되는 고양이/개 이미지 경로
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')


# 테스트에 사용되는 고양이/개 이미지 경로
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')




#신경망 생성

model_origin = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D(2,2),

  keras.layers.Dropout(0.25),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  keras.layers.Dropout(0.25),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  keras.layers.Dropout(0.25),

  tf.keras.layers.Flatten(),

  keras.layers.Dropout(0.25),

  tf.keras.layers.Dense(512, activation='relu'),

  keras.layers.Dropout(0.25),

  tf.keras.layers.Dense(1, activation='sigmoid')
])

model_origin.summary()







# 컴파일
from tensorflow.keras.optimizers import RMSprop
model_origin.compile(optimizer=RMSprop(lr=0.001),
            loss='binary_crossentropy',
            metrics = ['accuracy'])





# 이미지 데이터 전처리
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator( rescale = 1.0/255. )
train_datagen = ImageDataGenerator(rescale = 1.0/255.,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                  batch_size=20,
                                                  class_mode='binary',
                                                  target_size=(150, 150))
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                       batch_size=20,
                                                       class_mode  = 'binary',
                                                       target_size = (150, 150))








# 학습 과정 저장
history = model_origin.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=60,
                    validation_steps=50,
                    verbose=2)






# 훈련 과정 시각화
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()












# 모델 저장
from keras.models import load_model
model_origin.save('/content/drive/MyDrive/Colab Notebooks/dog_cat_model_D&G_chDP.h5')