from numpy.lib.function_base import average
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#import cv2
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler



HOME_TRAIN = False



BATCH_SIZE = 32
DROP_OUT_RATE = 0.2




dataset_info = pd.read_csv("meta_data_face_coor_K-Face.csv")
dataset_info



data_file_path = dataset_info[['file_path_x' , 'left' , 'right' , 'top' , 'bottom']]
gender = dataset_info['new_gender'].tolist()





file_path_train, file_path_val, y_train, y_val = train_test_split(data_file_path, gender, 
                                                                  test_size=0.25, 
                                                                  random_state=777, 
                                                                  stratify = gender)


# In[7]:


print( len(file_path_train) , len(y_train) , len(file_path_val) , len(y_val) )


train_left = file_path_train['left'].tolist()
train_right = file_path_train['right'].tolist()
train_top = file_path_train['top'].tolist()
train_bottom = file_path_train['bottom'].tolist()
file_path_train = file_path_train['file_path_x'].tolist()




val_left = file_path_val['left'].tolist()
val_right = file_path_val['right'].tolist()
val_top = file_path_val['top'].tolist()
val_bottom = file_path_val['bottom'].tolist()
file_path_val = file_path_val['file_path_x'].tolist()




def load_image( image_path , left , right , top , bottom , label ):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)   
    img = tf.image.crop_to_bounding_box( img , top , left, bottom - top , right - left )
    
    """
    output_image = tf.image.encode_png(img)
    file_name = tf.constant('./Ouput_image.png')
    file = tf.io.write_file(file_name, output_image)    
    """
    
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    
    label = tf.one_hot(label, 2)

    return img , label



train_dataset = tf.data.Dataset.from_tensor_slices( (file_path_train , 
                                                     train_left , 
                                                     train_right , 
                                                     train_top , 
                                                     train_bottom , 
                                                     y_train) )

val_dataset = tf.data.Dataset.from_tensor_slices( (file_path_val , 
                                                   val_left , 
                                                   val_right , 
                                                   val_top , 
                                                   val_bottom ,
                                                   y_val) )




train_dataset = train_dataset.shuffle(buffer_size=len(file_path_train))\
                                        .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                                        .batch(BATCH_SIZE)\
                                        .prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_dataset.shuffle(buffer_size=len(file_path_val))\
                            .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                            .batch(BATCH_SIZE)\
                            .prefetch(tf.data.experimental.AUTOTUNE)



ResNet50 = tf.keras.applications.resnet.ResNet50(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)




model= Sequential()

model.add( ResNet50 )

model.add( GlobalAveragePooling2D() ) 
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 
model.add( Dense(128, activation='relu') )
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 

#model.add( Dense(1, activation='sigmoid') )
model.add( Dense(2, activation='softmax') )  # One-Hot 인경우에 Softmax를 사용한다.



initial_learning_rate = 0.01

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * np.math.exp(-k*epoch)

lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)




log_dir = os.path.join('Logs')
CHECKPOINT_PATH = os.path.join('CheckPoints_K-Face_Gender_F1_Score')
tb_callback = TensorBoard(log_dir=log_dir)


cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                     #monitor='val_accuracy',
                     monitor='val_F1_metric',
                     save_best_only = True,
                     verbose = 1)


F1_metric = tfa.metrics.F1Score(num_classes=2 , average=None)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy' , F1_metric]
)



hist = model.fit(train_dataset,
                 validation_data=val_dataset,
                 callbacks=[lr_scheduler , cp , tb_callback],
                 epochs = 20,
                 verbose = 1 
)



