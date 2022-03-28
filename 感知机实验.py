import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split  #这里需要修改一下库名
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc

class Perceptron(object):

    # 初始化参数：学习率和迭代次数
    def __init__(self):
        self.learning_step = 0.00001    # 设置学习率(步长)为0.00001
        self.max_iteration = 5000       # 设置最大更新迭代次数为5000

    # 和predict函数共同组成预测函数，本函数的功能是实现预测并返回预测值
    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])  # 求w * x（这里是向量表示）的值
        if wx  > 0:       # 如果w * x > 0，则预测值为1
            return 1
        else:                          
            return 0      # 如果w * x < 0，则预测值为0

    # 训练函数
    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)   # 初始化w，b = 0。（对于w而言是指的0向量），至于为什么+1可以参见L35的注释

        correct_count = 0         # 正确分类的次数
        time = 0                  # 初始化迭代次数为0

        while time < self.max_iteration:  # 当更新迭代次数小于最大次数限制5000时，可以继续进行训练
            #随机梯度下降法求参数
            index = random.randint(0, len(labels) - 1)   # 从训练集中随机选取一个样本
            x = list(features[index])                    # 获取该样本的特征向量
            x.append(1.0)                                # 加偏置，这里我们把w和b放在一个矩阵中考虑，可以加速运算速度
            y = 2 * labels[index] - 1                    # 把y的值从0,1取值转换为-1，+1取值
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])  # 求w * x的值
            if wx * y > 0:                # 如果wx > 0,则该样本分类正确
                correct_count += 1        # 正确分类的次数+1
                if correct_count > self.max_iteration:
                    break          # 如果正确分类的次数大于最大更新迭代次数限制，则结束训练
                continue           # 否则进入下一样本的训练
            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])  # 发现误分类的样本，更新w

    #  # 与predict_(self, x)共同组成预测函数
    def predict(self,features):
        labels = []               # 创建labels数组用于存放预测值
        for feature in features:
            x = list(feature)     # list()：将元组转换为列表
            x.append(1)           # 参考L35的注解，我们要把w和b放在一个矩阵中，所以需要加一维度
            labels.append(self.predict_(x))   # 进行测试集的样本的预测
        return labels             # 返回预测值


def drawPR(P, R) :
    plt.figure("P-R curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(P, R)
    plt.show()
    return

def drawROC(fpr, tpr, auc):
    plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = "ROC curve (area = %0.2f)" % auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc = "lower right")
    plt.show()
    return 

if __name__ == '__main__':

    # 读取数据
    print('Start read data')

    time_1 = time.time()     # 获取当前时间记为time_1

    raw_data = pd.read_csv('C:/Users/sdlwq/Documents/WeChat Files/wxid_nzxxvy7wrx5q22/FileStorage/File/2022-03/train.csv', header=0)   # 加载数据集
    data = raw_data.values   # 读取数据集中的数据

    imgs = data[0::, 1::]    # 取数据集中从第2列至最后一列的所有数据
    labels = data[::, 0]     # 取数据集中第1列所有数据

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()    # 获取当前时间记为time_2
    print('read data cost ', time_2 - time_1, ' second', '\n')    # 输出读取数据所用时间

    # 训练数据
    print('Start training')
    p = Perceptron()       # 创建Perceptron类的对象为p
    p.train(train_features, train_labels)       # 对训练集数据进行训练

    time_3 = time.time()  # 获取当前时间记为time_3
    print('training cost ', time_3 - time_2, ' second', '\n')   # 输出训练数据所用时间

    # 对测试集数据进行预测
    print('Start predicting')
    test_predict = p.predict(test_features)     # 对测试集数据进行预测
    time_4 = time.time()  # 获取当前时间记为time_4
    print('predicting cost ', time_4 - time_3, ' second', '\n')  # 输出预测测试集数据所用时间

    # 计算该系统的预测正确率
    score = accuracy_score(test_labels, test_predict)
    print("The accruacy socre is ", score)   # 输出预测正确率
    #PR curve
    precision, recall, thresholds = precision_recall_curve(test_labels, test_predict)
    drawPR(precision, recall)
    #ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_predict)
    auc = metrics.auc(fpr, tpr)
    drawROC(fpr, tpr, auc)