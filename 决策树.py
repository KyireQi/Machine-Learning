import numpy as np
from math import log
from scipy.fftpack import ss_diff

from sklearn import datasets

def loadData():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    return datasets, labels

def calc_entropy(datasets) :
    lable_count = {}
    for dataset in datasets :
        lable = dataset[-1]
        if lable not in lable_count :
            lable_count[lable] = 0;
        lable_count[lable] += 1
    entropy = -sum([(p / len(datasets) * log(p / len(datasets), 2)) for p in lable_count.values()])
    return entropy
    
def get_conditional_entropy(datasets, index = 0) :
    feature_data = {}        
    # 计算H(D|A_index)
    for dataset in datasets :
        feature = dataset[index]
        if feature not in feature_data :
            feature_data[feature] = []
        feature_data[feature].append(dataset)
    condEntropy = sum([len(p) / len(datasets) * calc_entropy(p) for p in feature_data.values()])
    return condEntropy

def get_dis_entropy(entropy, condEntropy) :
    return entropy - condEntropy

def get_best_childtree(datasets, lables) :
    #获取信息熵
    entropy = calc_entropy(datasets)
    feature = []
    for index in range(len(datasets[0]) - 1) :
        #条件熵计算
        condEntropy = get_conditional_entropy(datasets, index);
        #互信息计算
        dis_entropy = get_dis_entropy(entropy, condEntropy);
        feature.append((index, dis_entropy));
        print("特征({})的信息增益为{:.3f}".format(lables[index], dis_entropy))
    #比较大小，取极大值
    best_feature = max(feature, key = lambda x : x[-1]);
    print("特征({})的信息增益最大，选择为当前节点特征".format(lables[best_feature[0]]))
    return best_feature

def info_gain_train(datasets, lables) :
    # 统计两种类别的数目
    lable_count = {}
    for feature in datasets:
        lable = feature[-1]
        if lable not in lable_count :
            lable_count[lable] = 0
        lable_count[lable] += 1;
    # 如果此时节点中只有一种类型的数据，证明到达了叶子节点，返回。
    if len(lable_count.keys()) == 1:
        key = list(lable_count.keys())[0];
        print("此时类别均为：{}".format(key))
        return 
    # 计算互信息，选取最佳分类特征
    best_feature = get_best_childtree(datasets, lables)
    #开始分类
    feature_data = {}
    for dataset in datasets :
        feature = dataset[best_feature[0]]
        if feature not in feature_data :
            feature_data[feature] = []
        feature_data[feature].append(dataset)
    
    for data in zip(feature_data.keys(), feature_data.values()) :
        print("当{}为{}".format(lables[best_feature[0]], data[0]))
        info_gain_train(data[1], lables)
    
if __name__ == "__main__" :
    datasets, lables = loadData();
    info_gain_train(datasets, lables);
