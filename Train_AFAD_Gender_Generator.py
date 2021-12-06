#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler


# In[2]:


data_generator = ImageDataGenerator(validation_split=0.25,
                                    rescale=1./255)


# In[3]:


BATCH_SIZE = 16
#BATCH_SIZE = 8 #회사 Main PC


# In[4]:


PATH = "./Dataset/AFAD-Full/Generator/"


# In[5]:


train_gen = data_generator.flow_from_directory(PATH,
                                               target_size = (224, 224),
                                               batch_size = BATCH_SIZE,  
                                               subset="training",
                                               class_mode = 'binary'
                                              )


# In[6]:


valid_gen = data_generator.flow_from_directory(PATH,
                                               target_size = (224, 224),
                                               batch_size = BATCH_SIZE,  
                                               subset="validation",
                                               class_mode = 'binary'
                                              )


# In[7]:


log_dir = os.path.join('Logs')
CHECKPOINT_PATH = os.path.join('CheckPoints')
tb_callback = TensorBoard(log_dir=log_dir)

cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='val_accuracy',
                     save_best_only = True,
                     verbose = 1)


# In[8]:


initial_learning_rate = 0.01

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * np.math.exp(-k*epoch)


# In[9]:


lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)


# In[10]:


ResNet50 = tf.keras.applications.resnet.ResNet50(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)

"""
EfficientNet = tf.keras.applications.efficientnet.EfficientNetB0(
    #weights='imagenet',
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)
"""


# In[11]:


#EfficientNet.trainable = False


# In[12]:


DROP_OUT_RATE = 0.25


# In[13]:


model= Sequential()

#model.add( EfficientNet )
model.add( ResNet50 )

model.add( GlobalAveragePooling2D() ) 
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 
model.add( Dense(512, activation='relu') )
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 
model.add( Dense(128, activation='relu') )
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 
model.add( Dense(1, activation='sigmoid') )


# In[14]:


model.summary()


# In[15]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


hist = model.fit(train_gen,
                 epochs = 50,
                 validation_data=valid_gen,
                 callbacks=[cp , tb_callback , lr_scheduler],
                 verbose = 1 
)


# In[ ]:





# In[ ]:




