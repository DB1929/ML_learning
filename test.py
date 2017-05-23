#-*-coding:utf-8 -*-
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import random
import io
import jieba

def get_dataset():
    data = []
    for root,dirs,files in os.walk(r'/Volumes/Files/WorkFile/Demo/test_ML/tokens/neg'):
        for file in files:
            realpath = os.path.join(root,file)
            with io.open(realpath, errors='ignore') as f:
                data.append((f.read(),'bad'))

    for root,dirs,files in os.walk(r'/Volumes/Files/WorkFile/Demo/test_ML/tokens/pos'):
        for file in files:
            realpath = os.path.join(root,file)
            with io.open(realpath, errors='ignore') as f:
                data.append((f.read(),'good'))
    random.shuffle(data)

    return data


data = get_dataset()

# print(data)

def train_and_test_data(data_):
    filesize = int(0.7*len(data_))#训练集和测试集的比例为7:3
    train_data = [each[0] for each in data_[:filesize]]
    train_tartget = [each[1] for each in data_[:filesize]]

    test_data = [each[0] for each in data_[filesize:]]
    test_tartget = [each[1] for each in data_[filesize:]]

    return train_data,train_tartget,test_data,test_tartget

train_data_,train_tartget_,test_data_,test_tartget_ = train_and_test_data(data)

nbc = Pipeline([
    ('vecr',TfidfVectorizer()),
    ('clf',MultinomialNB(alpha=1.0))
])

nbc.fit(train_data_,train_tartget_) #训练多项式模型贝叶斯分类器

print(test_data_)

predict = nbc.predict(test_data_) #在测试集上预测结果

count = 0

for left, right in zip(predict, test_tartget_):
    if left == right:
        count += 1
print(len(test_tartget_))
print(count)
print((float(count)/float(len(test_tartget_))))
result = (float(count)/float(len(test_tartget_)))
if result < 0.7:
    print "结果不太好"
else:
    print '口碑不错'



seg_list = jieba.cut("建议加载时使用展位图，加载完成后再懒加载页面内图片")
print("Full Mode: " + "/ ".join(seg_list))  # 全模式
