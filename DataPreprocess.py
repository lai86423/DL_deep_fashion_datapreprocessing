#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  # plt 用於顯示圖片
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
import pandas as pd
from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping
import random

early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        verbose=1,
                        mode='auto',
                        epsilon=0.0001)

def ReadFile(data_path, data):
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

def ReadFile_all(data_path, data):
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line)
    print(len(data))

def data_generator(data, targets, batch_size):
    batches = (len(data) + batch_size - 1)//batch_size
    while True:
        for i in range(batches):
            X = data[i*batch_size: (i+1)*batch_size]
            Y = targets[i*batch_size: (i+1)*batch_size]
            yield (X, Y)

# 資料路徑--------------------------------------------------------------------
base_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark'
tarintxt_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark\\Anno_fine\\train.txt'
tarincate_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark\\Anno_fine\\train_cate.txt'
base_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark'
testtxt_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark\\Anno_fine\\test.txt'
testcate_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark\\Anno_fine\\test_cate.txt'
tarintxt_all_path ='C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark\\allimg_train.txt'
tarincate_all_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark\\allimg_train_cate.txt'

# 製作訓練資料 標籤&資料集------------------------------------------------------
img_root = []
imgcate = []


# 讀取資料
trainPath = ReadFile(tarintxt_all_path, img_root)
trainCate = ReadFile(tarincate_all_path, imgcate)
# 重新排列資料
state = np.random.get_state()
np.random.shuffle(trainPath)
np.random.set_state(state)
np.random.shuffle(trainCate)

#設定input 維度
val_per = 0.05
dim1 = 128
dim2 = 128

# 分配訓練集及測試集比例------------------------------------
num = int((1 - val_per) * len(trainCate)) 
x_train = trainPath[:num]
y_train = trainCate[:num]
x_val = trainPath[num:]
y_val = trainCate[num:]


#設定測試資料集和對應標籤----------------------------------
val_num = len(x_val)
x_val_data = np.zeros((val_num, dim1, dim2, 3))
y_val_data = np.zeros((val_num))
non_exist = [] 

for i in range(val_num):
    if os.path.isfile(x_val[i]):
        img = cv2.imread(x_val[i])#讀圖
        img = cv2.resize(img, (dim1, dim2), interpolation=cv2.INTER_LINEAR)
        img = img_to_array(img)
        x_val_data[i] = img
        y_val_data[i] = y_val[i]
    else:
        non_exist.append(i)

# Delete non_exist   
x_val_data = np.delete(x_val_data, non_exist, axis=0)
y_val_data = np.delete(y_val_data, non_exist, axis=0)
# One Hot Encoding
y_val_data = np_utils.to_categorical(y_val_data)

print("val_len : ", x_val_data.shape)
np.save(os.path.join(base_path, 'x_val.npy'), x_val_data)
np.save(os.path.join(base_path, 'y_val.npy'), y_val_data)


# 設定訓練資料集----------------------------------------
data_num = len(x_train)
x_train_data = np.zeros((21600, dim1, dim2, 3))
y_train_data = np.zeros(21600)


k = 0   # 第k筆npy
file_cot = 0    # 儲存檔案數
for i in range(data_num):
    if os.path.isfile(x_train[i]):
        if y_train[i] != 0:
            img = cv2.imread(x_train[i])#讀圖
            img = cv2.resize(img, (dim1, dim2), interpolation=cv2.INTER_LINEAR)
            img = img_to_array(img)
            x_train_data[k] = img
            y_train_data[k] = y_train[i]
            k += 1

        if k == 21600:
            print("file_count", file_cot+1)
            # One Hot Encoding
            y_train_onehot = np_utils.to_categorical(y_train_data)
            print(f'x training data', x_train_data.shape)
            print(f'y training data', y_train_onehot.shape)
            np.save(os.path.join(base_path,'inputs' + str(file_cot + 1) + '.npy'), x_train_data)
            np.save(os.path.join(base_path,'labels' + str(file_cot + 1) + '.npy'), y_train_onehot)
            k = 0
            file_cot += 1

#設定訓練資料集和對應標籤
#x_train = np.zeros((data_num, dim1, dim2, 3))
#y_train = np.zeros((data_num))
# non_exist = []
# count = 0
# for i in range(data_num):
#     if os.path.isfile(img_root[i]):
#         img = cv2.imread(img_root[i])#讀圖
#         img = cv2.resize(img, (dim1, dim2), interpolation=cv2.INTER_LINEAR)
#         img = img_to_array(img)
#         x_train[i] = img
#     else:
#         count += 1
#         non_exist.append(i)

#     y_train[i] = imgcate[i]

# # Delete non_exist   
# x_train = np.delete(x_train, non_exist, axis=0)
# y_train = np.delete(y_train, non_exist, axis=0)

# # One Hot Encoding
# y_train = np_utils.to_categorical(y_train)

# print("training data resize ok!")
# print(f'x training data', x_train.shape)
# print(f'y training data', y_train.shape)

# # 製作測試資料 標籤&資料集------------------------------------------------------
# test_img_root = []
# test_imgcate = []

# # 讀取資料
# ReadFile(testtxt_path, test_img_root)
# ReadFile(testcate_path, test_imgcate)

# test_data_num = len(test_img_root)

# #設定測試資料集和對應標籤
# x_test = np.zeros((test_data_num, dim1, dim2, 3))
# y_test = np.zeros((test_data_num))
# non_exist = [] 
# count = 0
# for i in range(test_data_num):
#     if os.path.isfile(test_img_root[i]):
#         img = cv2.imread(test_img_root[i])#讀圖
#         img = cv2.resize(img, (dim1, dim2), interpolation=cv2.INTER_LINEAR)
#         img = img_to_array(img)
#         x_test[i] = img
#     else:
#         count += 1
#         non_exist.append(i)

#     y_test[i] = test_imgcate[i]

# # Delete non_exist   
# x_test = np.delete(x_test, non_exist, axis=0)
# y_test = np.delete(y_test, non_exist, axis=0)

# # One Hot Encoding
# y_test = np_utils.to_categorical(y_test)

# print("testing data resize ok!!")
# print(f'x testing data', x_test.shape)
# print(f'y testing data', y_test.shape)


