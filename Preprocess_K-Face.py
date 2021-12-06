import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

dir_path = "./Dataset/K-Face/Middle_Resolution"
data_file_path = []

for (root, directories, files) in tqdm(os.walk(dir_path)):
    for file in files:
        if '.jpg' in file:
            file_path = os.path.join(root, file)
            data_file_path.append( file_path )


print( len(data_file_path) )

meta_data = pd.DataFrame( data_file_path , columns=['file_path'])
meta_data.to_csv("meta_data_K-Face.csv",index=False)


"""
Folder 구조
ID 
  - S001 ~ S006 : 악세사리 착용
    1 : 악세사리 없음
    2 : 일반 안경
    3 : 뿔테안경
    4 : 선글라스
    5 : 모자
    6 : 모자 + 뿔테안경
"""