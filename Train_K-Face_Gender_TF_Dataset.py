import pandas as pd
import os
import numpy as np
#from six import b
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2

"""
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler
"""


HOME_TRAIN = False
BATCH_SIZE = 32

dataset_info = pd.read_csv("meta_data_for_train_K-Face.csv")
dataset_info


def gender_map( gender ):
    if gender == '남':
        return 1
    elif gender == '여':
        return 0


dataset_info['new_gender'] = dataset_info['성별'].map( gender_map )


data_file_path = dataset_info['file_path'].tolist()
gender = dataset_info['new_gender'].tolist()
#age = dataset_info['age'].tolist()

if HOME_TRAIN == False:
    data_file_path = [c.replace('f:/', 'C:/Users/csyi/Desktop/Age_Gender_Prediction/Dataset/K-Face/') for c in data_file_path]


img = cv2.imread( data_file_path[0] )

cv2.imshow("Test" , img)
cv2.waitKey()
cv2.destroyAllWindows()