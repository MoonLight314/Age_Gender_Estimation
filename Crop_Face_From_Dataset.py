import pandas as pd
import os
import numpy as np
#from tensorflow.keras.applications.efficientnet import EfficientNetB0
import cv2
import matplotlib.image as img
from tqdm import tqdm
import matplotlib.pyplot as plt





CONFIDENCE_FACE = 0.9




meta_info = pd.DataFrame()

for idx in range(5):
    path = "Dataset/Adience_Dataset/fold_{}_data.txt".format(idx)
    data = pd.read_csv(path, sep = "\t")
    
    meta_info = pd.concat([meta_info,data])


meta_info.reset_index(drop = True , inplace = True)

meta_info.drop(meta_info[meta_info['age'] == 'None'].index , inplace=True)
meta_info.drop(meta_info[meta_info['age'] == '(8, 23)'].index , inplace=True)



def preprocessing_age( age ):
    if age == '(8, 12)':
        return '(8, 13)'
    
    if age == '2':
        return '(0, 2)'
    
    if age == '3':
        return '(4, 6)'
    
    if age == '13':
        return '(8, 13)'
    
    if age == '22':
        return '(15, 20)'
    
    if age == '35' or age == '45' or age == '36' or age == '(38, 42)' or age == '(38, 48)' or age == '42':
        return '(38, 43)'
    
    if age == '34' or age == '23' or age == '(27, 32)' or age == '29' or age == '32':
        return '(25, 32)'
    
    if age == '55' or age == '56' or age == '46':
        return '(48, 53)'
    
    if age == '57' or age == '58':
        return '(60, 100)'
    
    return age



meta_info['new_age'] = meta_info['age'].apply( preprocessing_age )
meta_info.drop(meta_info[meta_info['gender'] == 'u'].index , inplace=True)
meta_info.dropna(axis=0, subset=['gender'], inplace=True)


meta_info.reset_index(drop = True , inplace = True)

user_id = meta_info['user_id'].astype('str').tolist()
original_image = meta_info['original_image'].astype('str').tolist()
face_id = meta_info['face_id'].astype('str').tolist()

modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


SIZE = 300

UP_EXPAND_FACTOR = 0.2
LOW_EXPAND_FACTOR = 0.05

crop_image_file_name = list()
gender = list()
age = list()


for idx in tqdm(range( len(user_id) )):
#for idx in range( 50 ):
    
    img_path = os.path.join( "Dataset" , "Adience_Dataset" , "faces" , user_id[idx] , "coarse_tilt_aligned_face." + face_id[idx] + "." + original_image[idx] )
    img = cv2.imread( img_path )
    rows, cols, channels = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 1.0,(SIZE, SIZE))#, (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()    
    
    conf = 0
    max_conf_dec = 0

    for detection in detections[0,0]:
        
        score = float(detection[2])

        if score > conf:
            conf = score
            max_conf_dec = detection
        

    if max_conf_dec[3] >= 1.00 or max_conf_dec[4] >= 1.00 or max_conf_dec[5] >= 1.00 or max_conf_dec[6] >= 1.00 or max_conf_dec[3] <= 0 or max_conf_dec[4] < 0 or max_conf_dec[5] <= 0 or max_conf_dec[6] <= 0:
        pass
    else:
        left = int(max_conf_dec[3] * cols)
        top = int(max_conf_dec[4] * rows)
        right = int(max_conf_dec[5] * cols)
        bottom = int(max_conf_dec[6] * rows)

        # 
        left = int(left - (left * UP_EXPAND_FACTOR))
        top = int(top - (top * UP_EXPAND_FACTOR))
        right = int(right + ( right * UP_EXPAND_FACTOR))
        bottom = int(bottom + ( bottom * LOW_EXPAND_FACTOR))
        
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        if right > cols:
            right = cols
        if bottom > rows:
            bottom = rows


        cropped = img[ top:bottom ,left:right]
        crop_img_path = os.path.join( "Dataset" , "Adience_Dataset" , "Crop_Faces_Rev_01" , str(idx) + ".jpg" )
        cv2.imwrite(crop_img_path , cropped)

        crop_image_file_name.append( crop_img_path )
        a = meta_info.loc[idx]
        gender.append( meta_info.loc[idx]['gender'] )
        age.append( meta_info.loc[idx]['new_age'] )


result = pd.DataFrame(list(zip(crop_image_file_name, gender , age)), columns=['file_path','gender','age'])
result.to_csv("crop_face_data_Rev_01.csv",index=False)