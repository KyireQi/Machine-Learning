from audioop import cross
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
# 下载中止词词集
import nltk
nltk.download()
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import roc_auc_score
# 将中止词存在stops里面，Python中对于集合的搜索比list快，所以我们转换成set操作
stops = set(stopwords.words("english"))

#该函数的作用是清洗我们的原始数据。
def DataWash(Input) :
    # 去掉html语法字段
    review_txt = BeautifulSoup(Input).get_text()
    # 使用正则表达式只保留字母部分
    review_txt = re.sub("[^a-zA-Z]", " ", review_txt)
    # 小写化所有的词，并转成词lists
    words = review_txt.lower().split()
    # 删除中止词
    meaning_words = [w for w in words if not w in stops]
    # 将处理好的数据合并成一个字符串，单词间用空格分隔开
    words = " ".join(meaning_words)
    return words

# 数据加载部分
train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)   # 加载训练集
test = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting = 3)    # 加载测试集

#训练集清洗部分
clean_train_reviews = []  # 存储清洗后的训练集数据
for i in tqdm(range(0, len(train["review"]))):  # 使用tqdm模块，以进度条的形式展示，我们处理的目标是所有电影的评价部分，即train['review']
    clean_train_reviews.append(DataWash(train['review'][i]))  # 对每一个电影的评论进行数据清洗。

# 特征处理部分，方法采用Bag of Words
# CountVectorizer是特征数值计算类，属于一种文本特征提取方法，对于每一个训练文本，
# 他只考虑每种词汇在该训练文本中出现的频率，会将文本中的词转换成为词频矩阵，并通过
# fit_transform 函数计算出各个词出现的频数。
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()  # 转换成ndarray类型方便处理
# print(train_data_features.shape)  # 25000 x 5000

# 引入朴素贝叶斯算法
from sklearn.naive_bayes import MultinomialNB as MNB
model_MNB = MNB()  # 创建MNB对象
model_MNB.fit(train_data_features, train["sentiment"])  # 进行朴素贝叶斯训练

# K折交叉验证
from sklearn.model_selection import cross_val_score
# 这里采用20折交叉验证，并通过ROC曲线的AUC大小评判打分
score = np.mean(cross_val_score(model_MNB, train_data_features, train["sentiment"], cv = 20, scoring = "roc_auc"))
print("20折交叉验证得分: ", score)

# 测试集清洗
clean_test_reviews = [] # 存储清洗后的测试集数据
for i in tqdm(range(0, len(test["review"]))):  # 使用tqdm模块，以进度条的形式展示，我们处理的目标是所有电影的评价部分，即train['review']
    clean_test_reviews.append(DataWash(test["review"][i])) # 对每一个电影的评论进行数据清洗。

# 特征处理，同上述。
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
test_data_features = vectorizer.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

#输出结果到csv文件中
result = model_MNB.predict(test_data_features)
output = pd.DataFrame(data = {"id":test["id"], "sentiment":result})
output.to_csv("NB_model.cvs", index = False, quoting = 3)
