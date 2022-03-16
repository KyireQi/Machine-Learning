#encoding=utf-8

import functools
import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    features = []
    hog = cv2.HOGDescriptor('C:/Users/sdlwq/Documents/WeChat Files/wxid_nzxxvy7wrx5q22/FileStorage/File/2022-03/hog.xml')
    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features,(-1,324))

    return features

def compare_personal(x, y) :
    return x[0] < y[0]

def Predict(testset,trainset,train_labels):

    predict = []
    count = 0
    for test_vec in testset:
        
        # print (count)
        # count += 1

        knn_list = []       # 当前k个最近邻居
        max_index = -1      # 当前k个最近邻居中距离最远点的坐标
        max_dist = 0        # 当前k个最近邻居中距离最远点的距离

        
        for i in range(k):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec - test_vec) #linalg.norm求范数，默认是二范数
            knn_list.append((dist,label))

        # 剩下的点
        for i in range(k,len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]

            dist = np.linalg.norm(train_vec - test_vec)         

            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]

            if dist < max_dist:
                knn_list[max_index] = (dist,label)
                max_index = -1
                max_dist = 0

        knn_list.sort(key = functools.cmp_to_key(compare_personal))
        cnt = 1
        max = 0
        max_label = 0
        num = {}
        for i in range(len(knn_list)) :
            if cnt <= 10:
                if knn_list[i][1] not in num :
                    num[knn_list[i][1]] = 0
                num[knn_list[i][1]] += 1
                if num[knn_list[i][1]] > max :
                    max = num[knn_list[i][1]]
                    max_label = knn_list[i][1]
                cnt += 1
            else : 
                break
        predict.append(max_label)

    return np.array(predict)

k = 10

if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('C:/Users/sdlwq/Documents/WeChat Files/wxid_nzxxvy7wrx5q22/FileStorage/File/2022-03/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    features = get_hog_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print ('read data cost ',time_2 - time_1,' second','\n')

    print ('Start training')
    print ('knn do not need to train')
    time_3 = time.time()
    print ('training cost ',time_3 - time_2,' second','\n')

    print ('Start predicting')
    test_predict = Predict(test_features,train_features,train_labels)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)