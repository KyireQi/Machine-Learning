#encoding=utf-8

from queue import PriorityQueue
import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV,cv_img) 
    return cv_img

def Train(trainset, train_labels):
    prior_probability = np.zeros(class_num)                         # 先验概率
    conditional_probability = np.zeros((class_num, feature_len, 2))   # 条件概率

    # 计算先验概率及条件概率
    for i in range(len(train_labels)):
        img = binaryzation(trainset[i])     # 图片二值化
        label = train_labels[i]
        prior_probability[label] += 1
        for j in range(feature_len) :
            conditional_probability[label][j][img[j]] += 1
    
    print(prior_probability)

    for i in range(class_num):
        for j in range(feature_len):
            conditional_probability[i][j][0] = (conditional_probability[i][j][0] + 1) / (prior_probability[i] + 2)
            conditional_probability[i][j][1] = (conditional_probability[i][j][1] + 1) / (prior_probability[i] + 2)

    prior_probability /= (len(train_labels) + 10)

    return prior_probability, conditional_probability

# 计算概率
def calculate_probability(img,label,prior_probability,conditional_probability):
    Py = prior_probability[label]
    probability = Py
    for i in range(feature_len) :
        probability *= conditional_probability[label][i][img[i]]
    return probability

def Predict(testset,prior_probability,conditional_probability):
    predict = []

    for img in testset:
        # 图像二值化
        img = binaryzation(img)

        max_label = 0
        max_probability = calculate_probability(img, 0, prior_probability,conditional_probability)

        for j in range(1,10):
            probability = calculate_probability(img, j, prior_probability,conditional_probability)

            if max_probability < probability:
                max_label = j
                max_probability = probability

        predict.append(max_label)

    return np.array(predict)


class_num = 10
feature_len = 784

if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print ('read data cost ',time_2 - time_1,' second','\n')

    print ('Start training')
    prior_probability, conditional_probability = Train(train_features,train_labels)
    time_3 = time.time()
    print ('training cost ',time_3 - time_2,' second','\n')

    print ('Start predicting')
    test_predict = Predict(test_features,prior_probability,conditional_probability)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)