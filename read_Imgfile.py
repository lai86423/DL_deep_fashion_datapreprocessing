import cv2
import os
import numpy as np
import pandas
tarintxt_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark\\Anno_fine\\train.txt'
tarincate_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark\\Anno_fine\\train_cate.txt'
img_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark\\img'
root_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion\\data\\Category and Attribute Prediction Benchmark'
img_root = []
imgcate = []
imgcombine = []

def ReadFiletoList(data_path, data):
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 

def WriteFile(filename, data):
    newf = open(filename, 'w')
    for i in range(len(data)):
        newf.write(data[i]+'\n')
    newf.close()

ReadFiletoList(tarintxt_path, img_root)
ReadFiletoList(tarincate_path, imgcate)

# train3.txt ------------------------------------------------------------------------
# for i in range(len(img_root)):
#     imgcombine.append((img_root[i], str(imgcate[i])))
# imgcombine.sort()
# print(imgcombine)
#WriteFile('train3.txt',imgcombine)

# 對應訓練資料夾 和 對應號碼 ----------------------------------------------------------
# (已處理train4.txt 路徑和類型合併 且用/隔開)
newtrain = []
ReadFiletoList(root_path + '\\' + 'train4.txt',newtrain)
file_item = []
cate_coin_key = []

#取出資料夾名稱和對應號碼
for i in range(len(newtrain)):
    file_item.append(newtrain[i].split('/'))
    cate_coin_key.append((file_item[i][1],file_item[i][3]))
#照字母排序
cate_coin_key.sort()

#刪除重複的項目
cate_coin_num2 = pandas.unique(cate_coin_key).tolist()

#存成Dictionary
cate_dict = {}
for i in range(len(cate_coin_num2)): 
    cate_dict[cate_coin_num2[i][0]] =cate_coin_num2[i][1]
#print(cate_dict)

# 讀所有圖資料夾與其內所有圖片 並將路徑寫入新檔案 ----------------------------------------------------------
# allFileList = os.listdir(img_path)
# new_img_root = open('img_all.txt', 'w')

# for i in range(len(allFileList)):
#    for filename in os.listdir(img_path + "\\" + allFileList[i]):
#         new_img_root.write('img' + '\\' + allFileList[i] + '\\' + filename + '\n')

# new_img_root.close()

new_imgroot = []
ReadFiletoList(root_path + '\\' + 'img_all.txt' , new_imgroot)
newfile_item = []
newtrain_cate = []
not_exist = []
newdata = []

# 取出資料夾名稱和對應號碼
for i in range(len(new_imgroot)):
    newfile_item.append(new_imgroot[i].split('/'))
    try:
        newtrain_cate.append(cate_dict[newfile_item[i][1]])
    except KeyError:
        not_exist.append((i, newfile_item[i][1]))
        newtrain_cate.append('0')
    newdata.append(new_imgroot[i] + ' ' + newtrain_cate[i])

print("-----------------------------------------")
print(len(newdata))
WriteFile('allimg_train_to_cate.txt', newdata)
WriteFile('allimg_train.txt', new_imgroot)
WriteFile('allimg_train_cate.txt', newtrain_cate)

#print(new_traincate)


