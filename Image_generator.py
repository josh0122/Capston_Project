#!/usr/bin/env python
# coding: utf-8

# In[37]:


import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img 
from keras.preprocessing.image import ImageDataGenerator
# 이미지 증식 
augGen=ImageDataGenerator(rescale=1./255,
                  brightness_range=[0, 1.1],

                  fill_mode='nearest')


# In[38]:


import os
path = 'C:/Users/IME/Desktop/New'
save = "C:/Users/IME/Desktop/New1"
names = os.listdir(path)


# In[39]:


i=0
for name in names:
    
    a = os.path.join(path , name )
    
    image = cv2.imread(a , cv2.IMREAD_GRAYSCALE)   
    
    x=img_to_array(image)
    x=x.reshape((1,)+x.shape)
    
    # 이미지 증식 
    z=0
    for batch in augGen.flow(x,batch_size=1, save_to_dir='C:/Users/IME/Desktop/New1', save_prefix="fake",save_format="jpg"):
        z+=1
        if z > 100:
            break

    i += 1

