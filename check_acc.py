import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from google.colab import files
from keras.preprocessing import image

# https://codetorial.net/tensorflow/classifying_the_cats_and_dogs.html







# 불러오rl
# 저 괄호안에 드라이브 마운드한 주소 넣으면 되는 부분
new_model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/dog_cat_model_drop.h5')









# 테스트 데이터 입력 확인

uploaded=files.upload()

for fn in uploaded.keys():

  path='/content/' + fn
  img=image.load_img(path, target_size=(150, 150))

  x=image.img_to_array(img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])

  classes = new_model.predict(images, batch_size=10)

#   print(classes[0])

  if classes[0]>0:
    print(fn + " is a dog")
  else:
    print(fn + " is a cat")