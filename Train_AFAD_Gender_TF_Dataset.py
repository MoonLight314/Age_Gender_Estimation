#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
from six import b
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler


BATCH_SIZE = 32

"""
MEMORY_LIMITS = 4096
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = MEMORY_LIMITS)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
"""

# In[2]:


dataset_info = pd.read_csv("meta_data_AFAD_dataset.csv")
dataset_info


# In[3]:


def gender_map( gender ):
    if gender == 'male':
        return 1
    elif gender == 'female':
        return 0


# In[4]:


dataset_info['new_gender'] = dataset_info['gender'].map( gender_map )
dataset_info.head()


# In[5]:


data_file_path = dataset_info['file_path'].tolist()
gender = dataset_info['new_gender'].tolist()
age = dataset_info['age'].tolist()


# In[6]:


"""
le_gender = LabelEncoder( ).fit_transform(gender)

gender = tf.keras.utils.to_categorical(le_gender, num_classes=2)
gender
"""


# In[7]:


dataset_info['new_gender'].value_counts(dropna=False)


# In[8]:


"""
le_age = LabelEncoder( ).fit_transform(age)

age = tf.keras.utils.to_categorical(le_age, num_classes=8)
age
"""


# In[ ]:





# In[9]:


file_path_train, file_path_val, y_train, y_val = train_test_split(data_file_path, gender, 
                                                                  test_size=0.25, 
                                                                  random_state=777, 
                                                                  stratify = gender)


# In[10]:


print( len(file_path_train) , len(y_train) )


# In[11]:


print( len(file_path_val) , len(y_val) )


# In[12]:


y_train


# In[ ]:





# In[13]:


# In[14]:


def load_image( image_path , label ):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img , label


# In[15]:


train_dataset = tf.data.Dataset.from_tensor_slices( (file_path_train , y_train) )
val_dataset = tf.data.Dataset.from_tensor_slices( (file_path_val , y_val) )


# In[16]:

"""
train_dataset = train_dataset.shuffle(buffer_size=len(file_path_train))\
                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(BATCH_SIZE)\
                .cache()\
                .prefetch(tf.data.experimental.AUTOTUNE)     #

val_dataset = val_dataset.shuffle(buffer_size=len(file_path_val))\
                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(BATCH_SIZE)\
                .cache()\
                .prefetch(tf.data.experimental.AUTOTUNE)    #
"""
train_dataset = train_dataset.shuffle(buffer_size=len(file_path_train))\
                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(BATCH_SIZE)\
                .prefetch(tf.data.experimental.AUTOTUNE)     #

val_dataset = val_dataset.shuffle(buffer_size=len(file_path_val))\
                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(BATCH_SIZE)\
                .prefetch(tf.data.experimental.AUTOTUNE)    #

# In[17]:


train_dataset.take(1)


# In[18]:


val_dataset.take(1)


# In[ ]:





# In[ ]:





# In[19]:


ResNet50 = tf.keras.applications.resnet.ResNet50(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)

"""
EfficientNet = tf.keras.applications.efficientnet.EfficientNetB0(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)
"""    


# In[20]:


#EfficientNet.trainable = False


# In[21]:


DROP_OUT_RATE = 0.2


# In[22]:


model= Sequential()

#model.add( EfficientNet )
model.add( ResNet50 )

model.add( GlobalAveragePooling2D() ) 
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 
model.add( Dense(128, activation='relu') )
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 

#model.add( Dense(2, activation='softmax') )
model.add( Dense(1, activation='sigmoid') )


# In[23]:


model.summary()


# In[ ]:





# In[24]:


initial_learning_rate = 0.01

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * np.math.exp(-k*epoch)


# In[25]:


lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)


# In[26]:


log_dir = os.path.join('Logs')
CHECKPOINT_PATH = os.path.join('CheckPoints')
tb_callback = TensorBoard(log_dir=log_dir)

cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                     monitor='val_accuracy',                     
                     save_best_only = True,
                     verbose = 1)


# In[ ]:





# In[28]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


hist = model.fit(train_dataset,
                 validation_data=val_dataset,
                 callbacks=[lr_scheduler , cp , tb_callback],
                 epochs = 20,
                 verbose = 1 
)


# In[ ]:




