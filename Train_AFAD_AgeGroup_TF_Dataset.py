#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os
import numpy as np
from six import b
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler


BATCH_SIZE = 32
DROP_OUT_RATE = 0.35
initial_learning_rate = 0.01
VALID_RATIO = 0.33



"""
EfficientNet = tf.keras.applications.efficientnet.EfficientNetB0(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)
"""    




def Age_Group_map( age ):
    if age >= 10 and age < 20:
        return '10~20'
    elif age >= 20 and age < 30:
        return '20~30'
    elif age >= 30 and age < 40:
        return '30~40'
    elif age >= 40 and age < 50:
        return '40~50'
    elif age >= 50 and age < 60:
        return '50~60'
    elif age >= 60:
        return 'Over 60'





def load_image( image_path , label ):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img , label




def lr_exp_decay(epoch, lr):
        k = 0.1
        return initial_learning_rate * np.math.exp(-k*epoch)



def Train():
    dataset_info = pd.read_csv("meta_data_AFAD_dataset.csv")    
    #dataset_info['age'].value_counts(dropna=False)
    
    # 
    dataset_info['age_group'] = dataset_info['age'].apply( Age_Group_map )
    #print( dataset_info['age_group'].value_counts(dropna=False) )

    data_file_path = dataset_info['file_path'].tolist()
    age = dataset_info['age_group'].tolist()

    le_age = LabelEncoder( ).fit_transform(age)
    age = tf.keras.utils.to_categorical(le_age, num_classes=6)

    file_path_train, file_path_val, y_train, y_val = train_test_split(data_file_path, age, 
                                                                  test_size = VALID_RATIO, 
                                                                  random_state=777, 
                                                                  stratify = age)

    #print(len(file_path_train) , len(file_path_val) , len(y_train) , len(y_val))

    train_dataset = tf.data.Dataset.from_tensor_slices( (file_path_train , y_train) )
    val_dataset = tf.data.Dataset.from_tensor_slices( (file_path_val , y_val) )

    train_dataset = train_dataset.shuffle(buffer_size=len(file_path_train))\
                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(BATCH_SIZE)\
                .prefetch(tf.data.experimental.AUTOTUNE)     #

    val_dataset = val_dataset.shuffle(buffer_size=len(file_path_val))\
                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(BATCH_SIZE)\
                .prefetch(tf.data.experimental.AUTOTUNE)    #

    ResNet50 = tf.keras.applications.resnet.ResNet50(
                weights=None,
                input_shape=(224, 224, 3),
                include_top=False
                )

    """
    DROP_OUT_RATE = 0.3 , 13 Epoch에서 Val = 0.65, 이후 내려감
    model= Sequential()
    model.add( ResNet50 )
    model.add( GlobalAveragePooling2D() ) 
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(512, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(64, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(6, activation='softmax') )    
    """

    """
    DROP_OUT_RATE = 0.3 , 10 Epoch에서 Val = 0.65, 이후 더 이상 올라가지 않음
    model= Sequential()
    model.add( ResNet50 )
    model.add( GlobalAveragePooling2D() ) 
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(1024, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(6, activation='softmax') )
    """

    """
    # DROP_OUT_RATE = 0.5 , Valid 비율 0.33
    # Train / Val 모두 0.68 / 0.64에서 더 이상 학습이 되지않음
    model= Sequential()
    model.add( ResNet50 )
    model.add( GlobalAveragePooling2D() ) 
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(1024, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(6, activation='softmax') )
    """

    # DROP_OUT_RATE = 0.35 , Valid 비율 0.33
    model= Sequential()
    model.add( ResNet50 )
    model.add( GlobalAveragePooling2D() ) 
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(1024, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(512, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(64, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(6, activation='softmax') )    

    model.summary()

    #
    lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)


    # 
    log_dir = os.path.join('Logs')
    CHECKPOINT_PATH = os.path.join('CheckPoints_AgeGroup')
    tb_callback = TensorBoard(log_dir=log_dir)

    cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                        monitor='val_accuracy',                     
                        save_best_only = True,
                        verbose = 1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    hist = model.fit(train_dataset,
                 validation_data=val_dataset,
                 callbacks=[lr_scheduler , cp , tb_callback],
                 #epochs = 100,
                 epochs = 3,
                 verbose = 1 
                 )


    hist_df = pd.DataFrame(hist.history) 
    
    return







def Generate_Models( drop_out_rate ):
    
    ResNet50 = tf.keras.applications.resnet.ResNet50(
                weights=None,
                input_shape=(224, 224, 3),
                include_top=False
                )

    models = []

    # 
    model_type_00 = Sequential()
    model_type_00.add( ResNet50 )
    model_type_00.add( GlobalAveragePooling2D() ) 
    model_type_00.add( Dropout( drop_out_rate ) ) 
    model_type_00.add( BatchNormalization() ) 
    model_type_00.add( Dense(512, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model_type_00.add( Dropout( drop_out_rate ) ) 
    model_type_00.add( BatchNormalization() ) 
    model_type_00.add( Dense(64, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model_type_00.add( Dropout( drop_out_rate ) ) 
    model_type_00.add( BatchNormalization() ) 
    model_type_00.add( Dense(6, activation='softmax') )

    models.append( model_type_00 )


    model_type_01= Sequential()
    model_type_01.add( ResNet50 )
    model_type_01.add( GlobalAveragePooling2D() ) 
    model_type_01.add( Dropout( drop_out_rate ) ) 
    model_type_01.add( BatchNormalization() ) 
    model_type_01.add( Dense(1024, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model_type_01.add( Dropout( drop_out_rate ) ) 
    model_type_01.add( BatchNormalization() ) 
    model_type_01.add( Dense(6, activation='softmax') )

    models.append( model_type_01 )


    #
    model_type_02 = Sequential()
    model_type_02.add( ResNet50 )
    model_type_02.add( GlobalAveragePooling2D() ) 
    model_type_02.add( Dropout( drop_out_rate ) ) 
    model_type_02.add( BatchNormalization() ) 
    model_type_02.add( Dense(1024, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model_type_02.add( Dropout( drop_out_rate ) ) 
    model_type_02.add( BatchNormalization() ) 
    model_type_02.add( Dense(6, activation='softmax') )

    models.append( model_type_02 )


    #
    model_type_03 = Sequential()
    model_type_03.add( ResNet50 )
    model_type_03.add( GlobalAveragePooling2D() ) 
    model_type_03.add( Dropout( drop_out_rate ) ) 
    model_type_03.add( BatchNormalization() ) 
    model_type_03.add( Dense(1024, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model_type_03.add( Dropout( drop_out_rate ) ) 
    model_type_03.add( BatchNormalization() ) 
    model_type_03.add( Dense(512, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model_type_03.add( Dropout( drop_out_rate ) ) 
    model_type_03.add( BatchNormalization() ) 
    model_type_03.add( Dense(64, activation='relu' , kernel_initializer=tf.keras.initializers.HeUniform()) )
    model_type_03.add( Dropout( drop_out_rate ) ) 
    model_type_03.add( BatchNormalization() ) 
    model_type_03.add( Dense(6, activation='softmax') )    

    models.append( model_type_03 )

    

    return models








def HyperParamSearch():
    dataset_info = pd.read_csv("meta_data_AFAD_dataset.csv")    

    dataset_info['age_group'] = dataset_info['age'].apply( Age_Group_map )

    data_file_path = dataset_info['file_path'].tolist()
    age = dataset_info['age_group'].tolist()

    le_age = LabelEncoder( ).fit_transform(age)
    age = tf.keras.utils.to_categorical(le_age, num_classes=6)

    lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)

    # Test할 Hyper Parameter 값들
    val_ratios = [0.25 , 0.3 , 0.35 , 0.4]
    drop_out_rates = [0.2 , 0.3 , 0.4 , 0.5]

    # Epoch
    EPOCH = 20

    ##
    for drop_out in drop_out_rates:

        models = Generate_Models( drop_out )

        for val_ratio in val_ratios:

            # Making Datasets
            file_path_train, file_path_val, y_train, y_val = train_test_split(data_file_path, age, 
                                                                    test_size = val_ratio, 
                                                                    random_state=777, 
                                                                    stratify = age)

            #print(len(file_path_train) , len(file_path_val) , len(y_train) , len(y_val))

            train_dataset = tf.data.Dataset.from_tensor_slices( (file_path_train , y_train) )
            val_dataset = tf.data.Dataset.from_tensor_slices( (file_path_val , y_val) )

            train_dataset = train_dataset.shuffle(buffer_size=len(file_path_train))\
                        .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                        .batch(BATCH_SIZE)\
                        .prefetch(tf.data.experimental.AUTOTUNE)     #

            val_dataset = val_dataset.shuffle(buffer_size=len(file_path_val))\
                        .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                        .batch(BATCH_SIZE)\
                        .prefetch(tf.data.experimental.AUTOTUNE)    #

            for idx,model in enumerate(models):
                
                tf.keras.backend.clear_session()

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                hist = model.fit(train_dataset,
                            validation_data=val_dataset,
                            callbacks=[ lr_scheduler ],
                            epochs = EPOCH,
                            verbose = 1 
                            )


                hist_df = pd.DataFrame(hist.history)
                hist_file_name = "History_Model_Type_{0}_DropOut_{1}_Val_Ratio_{2}.csv".format(idx,drop_out,val_ratio)
                hist_df.to_csv( hist_file_name , index=False)
    
    
    
    return





if __name__== '__main__':
    #Train()
    HyperParamSearch()