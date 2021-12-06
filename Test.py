import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import cv2


MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9
MARGIN_RAIO = 0.2

HOME_TRAIN = False

net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )

filename = []
left_list = []
right_list = []
top_list = []
bottom_list = []


meta_data = pd.read_csv("meta_data_for_train_K-Face.csv")
file_path = meta_data['file_path'].tolist()

if HOME_TRAIN == False:
    file_path = [c.replace('f:/', 'C:/Users/Moon/Desktop/Age_Gender_Prediction/Dataset/') for c in file_path]


for file in tqdm(file_path):

    img = cv2.imread(file)
    rows, cols, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1.0, (SIZE, SIZE))  # , (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    a = detections[0, 0]    
    i = np.argmax(a[:,2])

    if i != 0:
        print(file , "Max index is not 0")

    if a[i,2] < CONFIDENCE_FACE:
        print(file , "Low CONFIDENCE_FACE" , a[i,2])

    """
    for detection in detections[0, 0]:

        score = float(detection[2])

        if score > CONFIDENCE_FACE:

            if detection[3] >= 1.00 or detection[4] >= 1.00 or detection[5] >= 1.00 or detection[6] >= 1.00 or detection[3] <= 0 or detection[4] < 0 or detection[5] <= 0 or detection[6] <= 0:
                filename.append(np.NaN)
                left_list.append( np.NaN )
                right_list.append( np.NaN )
                top_list.append( np.NaN )
                bottom_list.append( np.NaN )
                print("NaN")
            else:
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)

                left = left - int((right - left) * MARGIN_RAIO)
                top = top - int((bottom - top) * MARGIN_RAIO)
                right = right + int((right - left) * MARGIN_RAIO)
                bottom = bottom + int((bottom - top) * MARGIN_RAIO / 2)

                if left < 0:
                    left = 0

                if right > cols:
                    right = cols

                if top < 0:
                    top = 0

                if bottom > rows:
                    bottom = rows
                    
                filename.append(file)
                left_list.append( left )
                right_list.append( right )
                top_list.append( top )
                bottom_list.append( bottom )
        else:
            print("NaN " , file , score)
        """