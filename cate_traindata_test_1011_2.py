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

# with open(tarintxt_path, 'r') as f:
#     for line in f:
#         img_root.append(line[:-1])
# print(len(img_root))  

# with open(tarincate_path, 'r') as f:
#     for line in f:
#         imgcate.append(line[:-1])
# print(len(imgcate))

# 讀取資料
ReadFile(tarintxt_all_path, img_root)
ReadFile(tarincate_all_path, imgcate)

#設定input 維度
val_per = 0.2
dim1 = 128
dim2 = 128
data_num = len(img_root)

#設定訓練資料集和對應標籤
x_train = np.zeros((data_num, dim1, dim2, 3))
y_train = np.zeros((data_num))
non_exist = []
count = 0
for i in range(data_num):
    if os.path.isfile(img_root[i]):
        img = cv2.imread(img_root[i])#讀圖
        img = cv2.resize(img, (dim1, dim2), interpolation=cv2.INTER_LINEAR)
        img = img_to_array(img)
        x_train[i] = img
    else:
        count += 1
        non_exist.append(i)

    y_train[i] = imgcate[i]

# Delete non_exist   
x_train = np.delete(x_train, non_exist, axis=0)
y_train = np.delete(y_train, non_exist, axis=0)

# One Hot Encoding
y_train = np_utils.to_categorical(y_train)

print("training data resize ok!")
print(f'x training data', x_train.shape)
print(f'y training data', y_train.shape)

# 製作測試資料 標籤&資料集------------------------------------------------------
test_img_root = []
test_imgcate = []

# 讀取資料
ReadFile(testtxt_path, test_img_root)
ReadFile(testcate_path, test_imgcate)

test_data_num = len(test_img_root)

#設定測試資料集和對應標籤
x_test = np.zeros((test_data_num, dim1, dim2, 3))
y_test = np.zeros((test_data_num))
non_exist = [] 
count = 0
for i in range(test_data_num):
    if os.path.isfile(test_img_root[i]):
        img = cv2.imread(test_img_root[i])#讀圖
        img = cv2.resize(img, (dim1, dim2), interpolation=cv2.INTER_LINEAR)
        img = img_to_array(img)
        x_test[i] = img
    else:
        count += 1
        non_exist.append(i)

    y_test[i] = test_imgcate[i]

# Delete non_exist   
x_test = np.delete(x_test, non_exist, axis=0)
y_test = np.delete(y_test, non_exist, axis=0)

# One Hot Encoding
y_test = np_utils.to_categorical(y_test)

print("testing data resize ok!!")
print(f'x testing data', x_test.shape)
print(f'y testing data', y_test.shape)

# 分配訓練集及測試集比例---------------------------------------------------------
num = int((1 - val_per) * data_num)
x_val = x_train[num:]
y_val = y_train[num:]
x_train = x_train[:num]
y_train = y_train[:num]

# Model -----------------------------------------------------------------------
model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#freeze some layers
for layer in model_resnet.layers[:-12]:
     # 6 - 12 - 18 have been tried. 12 is the best.
     layer.trainable = False
#model_resnet.trainable = False

#build the category classification branch in the model
x = model_resnet.output
x = layers.Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
y = Dense(49, activation='softmax', name='img')(x)

#create final model by specifying the input and outputs for the branches
final_model = Model(inputs=model_resnet.input, outputs=y)

#print(final_model.summary())

#opt = SGD(lr=0.001, momentum=0.9, nesterov=True)
opt = Adam(learning_rate=0.01)
final_model.compile(optimizer=opt,loss={'img':'categorical_crossentropy'},
                    metrics={'img':['accuracy','top_k_categorical_accuracy']}) #default:top-5

# Loading the data-------------------------------------------------------------
train_datagen = ImageDataGenerator(rotation_range=30.,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True)
test_datagen = ImageDataGenerator()

# 設定超參數HyperParameters
batch_size = 12
epochs = 50

history = final_model.fit_generator(
    generator=data_generator(x_train, y_train, batch_size),
    steps_per_epoch=(len(x_train) + batch_size - 1) // batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, rlr]
)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)
