#encoding = utf-8

import functools
import pandas as pd
import numpy as np
import cv2
import random
import time
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    # 记录图像HOG特征
    features = []
    # 调用Opencv的HOG描述子，返回一个HOGDescriptor类
    hog = cv2.HOGDescriptor('C:/Users/sdlwq/Documents/WeChat Files/wxid_nzxxvy7wrx5q22/FileStorage/File/2022-03/hog.xml')
    # 获得训练集图像的HOG特征
    for img in trainset:
        img = np.reshape(img,(28,28))   # 将每一个特征向量（1*784）改为28*28的矩阵形式
        cv_img = img.astype(np.uint8)   # 数据类型转换为uint8类型
        hog_feature = hog.compute(cv_img) # 计算给定cv_image的HOG描述符（特征向量）
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature) # 将HOG特征向量加入到features中

    features = np.array(features)  # list转换称ndarray
    features = np.reshape(features,(-1,324)) # 这里使用了reshape的模糊控制，固定为324列，而行数自动计算好
    # 返回得到的features
    return features

#自定义排序函数
def compare_personal(x, y) :
    return x[0] < y[0]

#预测函数
def Predict(testset,trainset,train_labels):
    
    predict = []  # 预测的结果list
    for test_vec in testset:

        knn_list = []       # 当前k个最近邻居
        max_index = -1      # 当前k个最近邻居中距离最远点的坐标
        max_dist = 0        # 当前k个最近邻居中距离最远点的距离

        for i in range(k):  # 先加入k个点作为最近的k个邻居
            label = train_labels[i]  # 获得标记值
            train_vec = trainset[i]  # 获取训练数据集的特征值向量
            dist = np.linalg.norm(train_vec - test_vec) #linalg.norm求范数，默认是二范数
            knn_list.append((dist, label))  # 加入到K个最近邻居中

        # 处理剩下的点
        for i in range(k,len(train_labels)):
            #获取数据
            label = train_labels[i]
            train_vec = trainset[i]
            #计算和当前测试点的距离
            dist = np.linalg.norm(train_vec - test_vec)    

            if max_index < 0:  # max_index = -1有两种意义：首先是表示这是第一次循环，没有计算max_dist；还有一个意义是表示截止到上一个点，还没有出现满足加入knn_list条件的点出现
                for j in range(k):   # 计算得到现有的k个点中距离测试点最远的点。
                    if max_dist < knn_list[j][0]:
                        max_index = j  # 记录点的坐标
                        max_dist = knn_list[max_index][0]  # 更新最大距离

            if dist < max_dist:   # 意味着这个点更优，应该加入到knn_list中
                knn_list[max_index] = (dist,label)  # 替换最远点
                max_index = -1  # 重置max_index，为下一次更新做准备
                max_dist = 0 # 重置max_dist

        knn_list.sort(key = functools.cmp_to_key(compare_personal))  # 将knn_list中的数据排序，按照dist从小到大排序
        cnt = 1  # 记录投票表决人数
        max = 0  # 记录最多的投票人数
        max_label = 0  # 记录获得最多票的类型
        num = {}  # 用于记录不同类型获得的票数
        for i in range(len(knn_list)) : # 开始投票
            if cnt <= k: # 如果投票人数不超过k
                if knn_list[i][1] not in num :  # 如果第一次投，初始化为0
                    num[knn_list[i][1]] = 0
                num[knn_list[i][1]] += 1  # 记录投票票数
                if num[knn_list[i][1]] > max : # 更新最大值
                    max = num[knn_list[i][1]]
                    max_label = knn_list[i][1]
                cnt += 1  # 投票人数+1
            else :  # 投票人数超过k，跳出
                break
        predict.append(max_label)  # 记录结果

    return np.array(predict) # 返回ndarray类型的结果

def drawPR(P, R) :  #画出PR曲线
    plt.figure("P-R curve") #定义窗口名
    plt.title('Precision/Recall Curve') # 图像标题
    plt.xlabel('Recall') #x轴名称
    plt.ylabel('Precision') #y轴名称
    plt.plot(P, R) # 使用pyplot的plot方法作图
    plt.show() #显示图像
    return

def drawROC(fpr, tpr, auc):
    plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = "ROC curve (area = %0.2f)" % auc) # 传入参数分别是：假正例率，真正例率，线条宽度，标签（加入了AUC的值）
    plt.xlim([0.0, 1.0]) # 设置x轴的范围
    plt.ylim([0.0, 1.05]) # 设置y的范围
    plt.title('ROC Curve') # 设置标题
    plt.xlabel('FPR') # 设置x轴名称
    plt.ylabel('TPR') # 设置y轴名称
    plt.legend(loc = "lower right") # 标签位置选择右下
    plt.show() # 显示图像
    return 

k = 10  # 定义K的大小

if __name__ == '__main__':

    print ('Start read data')
    # 记录开始读取数据的时间
    time_1 = time.time()
    # Pandoc读入数据
    raw_data = pd.read_csv('C:/Users/sdlwq/Documents/WeChat Files/wxid_nzxxvy7wrx5q22/FileStorage/File/2022-03/train.csv',header=0)
    # 获取数据部分
    data = raw_data.values
    # 将特征部分和标记部分利用切片分开
    imgs = data[0::,1::]
    labels = data[::,0]
    #调用get_hog_features函数获得图像的HOG特征
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
    # 开始预测，返回结果记录给test_predict
    test_predict = Predict(test_features,train_features,train_labels)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n')
    # 计算精确度
    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)

    #PR curve
    precision, recall, thresholds = precision_recall_curve(test_labels, test_predict) # 调用precision_recall_curve方法，返回正确率、召回率和阈值集合
    drawPR(precision, recall) # 画出PR图
    #ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_predict) # 调用metrics.roc_curve方法，返回fpr,tpr和阈值集合
    auc = metrics.auc(fpr, tpr) # 调用auc方法返回AUC的值
    drawROC(fpr, tpr, auc)