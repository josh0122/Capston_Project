#!/usr/bin/env python
# coding: utf-8

# In[31]:


import argparse
import glob
import numpy as np
import os.path as path
import scipy.misc
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K


K.set_image_data_format('channels_first')

## load modules
import cv2 as cv
import matplotlib.pyplot as plt
import os, time  
import numpy as np 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
print(tf.__version__)


# In[32]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
from keras.preprocessing import image
from PIL import Image


# In[33]:


# 압축해제된 데이터 경로를 찾아 복사해서 붙여넣어주세요
src = 'C:/Users/user/Desktop/new_start/'

img_size = 150  # 우리가 넣을려고 하는 이미지의 사이즈
channels = 1       # 1= 회색조 / 3= 다색조 
noise_dim = 100 # gan에 입력되는 noise에 대한 dimension # 기존값은 100이었음.


#이미지 읽기 

def img_read(src,file):
    img = cv.imread(src+file, cv.IMREAD_GRAYSCALE) #  cv.IMREAD_GRAYSCALE 추가하여 shape 5개로 뜨던것 4개로 줄임
    return img
def get_data():
    #src 경로에 있는 파일 명을 저장합니다. 
    files = os.listdir(src)
    X = []  

    # 경로와 파일명을 입력으로 넣어 확인하고 
    # 데이터를 255로 나눠서 0~1사이로 정규화 하여 X 리스트에 넣습니다. 

    for file in files:
      
        X.append((img_read(src,file)-127.5)/127.5) 

    # Train set(80%), Test set(20%)으로 나누기 
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=1,shuffle=True)
        
    # (x, 56, 56, 1) 차원으로 맞춰줌 
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)


    return X_train, X_test

# 데이터 셋 불러옴 (이미지만 필요해서 y 라벨 필요 없음)
X_train, X_test = get_data()
print("X_train.shape = {}".format(X_train.shape))
print("X_test.shape = {}".format(X_test.shape))


# In[34]:


# images 확인용
fig = plt.figure(figsize=(20,10))
nplot = 5
for i in range(1,nplot):
    ax = fig.add_subplot(1,nplot,i)
    ax.imshow(X_train[i, :, :, 0],cmap = 'gray')
plt.show()


# In[35]:


# ---------------------
#  Generator 모델 구성 (input : noise / output : image)
# ---------------------    
 
def build_generator():
    model = Sequential()
   
    model.add(layers.Dense(256, use_bias=False, input_shape=(100,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod(150*150), activation='tanh')) # Tanh 함수는 -1과 1사이의 값을 출력함
    model.add(layers.Reshape((150,150,1)))
    # noise 텐서 생성, model에 noise 넣으면 이미지 나옴
    noise = Input(shape=(100,))
    img = model(noise)
    model.summary()
    return Model(noise,img) 


# In[36]:


# 아직 훈련을 시키지 않은 G모델과 D모델을 사용해보면 아래와 같은 출력을 보입니다. 

#출처: https://dataplay.tistory.com/39 [데이터 놀이터]
# Optimizer
optimizer = Adam(0.0002, 0.8)

# generator 모델 생성과 컴파일(loss함수와 optimizer 설정)
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# 노이즈 만들어서 generator에 넣은 후 나오는 이미지 출력 (확인용)
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

# plot_model(generator, show_shapes=True)


# In[37]:


# ---------------------
#  Discriminator 모델 구성 (input : image / output : 판별값(0에서 1사이의 숫자))
# ---------------------   
def build_discriminator():
    model = tf.keras.Sequential()
    img_shape = (img_size, img_size, channels)
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.summary()
    # 이미지 들어갈 텐서 생성, model에 넣으면 판별값 나옴
    img = Input(shape=img_shape)
    validity = model(img)
        
    return Model(img, validity) 


# In[38]:


# discriminator 모델 생성과 컴파일(loss함수와 optimizer 설정, accaracy 측정)
discriminator = build_discriminator()
discriminator.compile(loss = 'binary_crossentropy',optimizer = optimizer, metrics   = ['accuracy'])

# image를 discriminator에 넣었을 때 판별값 나옴 (예시. 확인용)
decision = discriminator(generated_image)
print (decision)


# In[39]:


# Combined Model
# 랜덤으로 만든 이미지로부터 학습해서 새로운 이미지를 만들어내는 generator의 데이터를 discriminator가 분류. 

z = layers.Input(shape=(100,), name="noise_input")
img = generator(z)

# 모델을 합쳐서 학습하기 때문에 발란스 때문에 discriminator는 학습을 꺼둠. 우리는 generator만 학습
discriminator.trainable = False

# discriminator에 이미지를 입력으로 넣어서 진짜이미지인지 가짜이미지인지 판별
valid = discriminator(img)

# generator와 discriminator 모델 합침. (노이즈가 인풋으로 들어가서 판별결과가 아웃풋으로 나오게)
# discriminator를 속이도록 generator를 학습
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)
combined.summary()


# In[40]:


def train(epochs, batch_size=64, sample_interval=500):
              
        # 정답으로 사용 할 매트릭스. valid는 1, fake는 0
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        for epoch in range(epochs):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # batch_size만큼 이미지와 라벨을 랜덤으로 뽑음
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            
            # Sample noise 생성(batch_size만큼)
            noise = np.random.normal(0, 1, (batch_size, 100))
            
            # noise를 generator에 넣어서 fake image 이미지 생성
            gen_imgs = generator.predict(noise)
            
            # discriminator를 학습함. 진짜 이미지는 1이 나오게, 가짜 이미지는 0이 나오게
            # discriminator가 이미지를 판별한 값과 valid와 fake가 
            # 각각 같이 들어가서 binary_crossentropy으로 계산되어 업데이트함.
            d_loss_real = discriminator.train_on_batch(imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            
            # real을 넣었을 때와 fake를 넣었을 때의 discriminator의 loss값과 accracy값의 평균을 구함.
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # noise 생성
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            
            # noise가 들어가서 discriminator가 real image라고 판단하도록 generator를 학습
            g_loss = combined.train_on_batch(noise, valid)

            history.append({"D":d_loss[0],"G":g_loss})

            # sample_interval(1000) epoch 마다 loss와 accuracy와 이미지 출력
            if epoch % sample_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                sample_images(epoch)
                
        return(history)


# In[62]:


# 이미지 저장
def sample_images(epoch):
       
       noise = np.random.normal(0, 1, (64, noise_dim))

       gen_imgs = generator.predict(noise)

       
       # Rescale images 0 - 1
       gen_imgs = 0.5 * gen_imgs + 0.5

       
       cnt = 0
       for i in range(0,63):
           
           plt.imshow(gen_imgs[cnt, :,   :,   0], cmap='gray')
           plt.axis('off')
           cnt += 1
           plt.savefig("C:/Users/user/Desktop/new_start_gan/gan10/%d.jpg" % epoch, bbox_inches='tight',pad_inches = 0, dpi=49.9)
           plt.close()
           



# In[63]:


# GAN 실행 # 150size #lr= 0.0002 beta 0.8
history=train(epochs=30000, batch_size=128, sample_interval=500) # sample_interval : 제대로 훈련되고 있는지 sample 체크 간격 (출력) 
# 이 두 성취도는 위 모델 그림의 fake-result 와 real-result 로 다시 나타낼 수 있습니다.

#분류기의 성능(성취도) d_loss 는 real-result 는 높을수록, fake-result 는 낮을수록 좋습니다.

#생성기의 성능(성취도) g_loss 는 fake-result 가 높을수록 좋습니다.


# In[36]:


#대부분의 딥러닝에서 loss는 말그대로 '손실'을 의미하기 때문에 0에 가까울 수록 좋습니다. 하지만 GAN은 조금 다릅니다. D모델이 하는 역할은 진짜와 가짜를 구별하는 것입니다. 
#그런데 만약 D모델의 loss가 0이 됐다면, 
#진짜와 가짜를 완벽히 구별 할 수 있다는 의미입니다. 

#출처: https://dataplay.tistory.com/39 [데이터 놀이터]
# summarize history for loss # 1 
import pandas as pd 
hist = pd.DataFrame(history)
plt.figure(figsize=(10,5))
for colnm in hist.columns:
    plt.plot(hist[colnm],label=colnm)
plt.legend()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()


# In[105]:


generated_image = generator(tf.random.normal([1, noise_dim])) # 1

fig1 = plt.gca()
fig1.axes.xaxis.set_visible(False)
fig1.axes.yaxis.set_visible(False)
fig1 = plt.gcf()
plt.imshow(tf.reshape(generated_image, shape = (150,150)), cmap='gray')

