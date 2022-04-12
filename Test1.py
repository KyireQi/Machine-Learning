from audioop import cross
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import nltk
nltk.download()
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import roc_auc_score

def DataWash(Input) :
    # 去掉html
    review_txt = BeautifulSoup(Input).get_text()
    # 正则表达式
    review_txt = re.sub("[^a-zA-Z]", " ", review_txt)
    # 小写化所有的词，并转成词lists
    words = review_txt.lower().split()
    # 删除停用词
    stops = set(stopwords.words("english"))
    meaning_words = [w for w in words if not w in stops]
    words = " ".join(meaning_words)
    return words

# 数据加载
train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting = 3)

#训练集清理
clean_train_reviews = []
for i in tqdm(range(0, len(train["review"]))):
    clean_train_reviews.append(DataWash(train['review'][i]))

# 特征处理
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
print(train_data_features.shape)  # 25000 x 5000

# 引入朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB as MNB
model_MNB = MNB()
model_MNB.fit(train_data_features, train["sentiment"])

# K折交叉验证
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(model_MNB, train_data_features, train["sentiment"], cv = 20, scoring = "roc_auc"))
print("20折交叉验证得分: ", score)

# 测试集清理
clean_test_reviews = []
for i in tqdm(range(0, len(test["review"]))):
    clean_test_reviews.append(DataWash(test["review"][i]))

# 特征处理
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
test_data_features = vectorizer.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

#输出结果
result = model_MNB.predict(test_data_features)
print(type(result))
output = pd.DataFrame(data = {"id":test["id"], "sentiment":result})
print(type(output))
output.to_csv("NB_model.cvs", index = False, quoting = 3)
