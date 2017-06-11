# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:49:32 2017
scikit-learn 估计器  欧式距离  近邻算法
@author: Tywin
"""

import os
import numpy as np
import csv

from sklearn.cross_validation import train_test_split

from sklearn.neighbors import KNeighborsClassifier #导入K近邻分类器

from sklearn.cross_validation import cross_val_score

from matplotlib import pyplot as plt

data_filename =os.path.join("ionosphere.data") #加载数据集

# 创建Numpy数组X，Y存放数据集。
X = np.zeros((351,34),dtype = "float")
Y = np.zeros((351,),dtype = "bool")

#用csv导入数据集文件

with open(data_filename,'r') as input_file:
    reader = csv.reader(input_file)
    
# 遍历每行数据，每行数据代表一组测量结果，称之为数据集中的一个个体
    for i, row in enumerate(reader):
        #获取每一个个体前34个值，该数据集每行有35个值，前34为天线采集的数据，最后一个值不是'g'就是'b' 表述数据的好坏
        data = [float(datum) for datum in row[:-1]] 
        X[i] = data
        #最后获取每个给最后一个表示类别的值，把字幕转化为数字，如果类别为'g'，值为1 否则为0
        Y[i] = row[-1] == 'g'
        #目前，数据集读取到数组X中，类别读入了数组Y中。
        
X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state = 14)

#实例化K紧邻分析器
estimator = KNeighborsClassifier()

#估计器创建好后，接下来就要用训练数据进行训练，K近邻估计器分析训练集中的数据，比较待分类的新数据点和训练集中的数据，找到新数据点的紧邻
estimator.fit(X_train,y_train)

#接着，用测试集测试算法，评估训练完成的模型在测试集上的表现
y_predicted = estimator.predict(X_test)

accuracy = np.mean(y_test == y_predicted) * 100

print("精确性是 {0:.1f}%".format(accuracy))

#运行算法,交叉验证避免一次性测试所带来的问题。每次切分时，都要保证这次得到的训练集和测试集与上次的不一样，还要确保每条数据都只能用来测试一次
'''
交叉验证算法描述:
1.将整个大数据集分为几个部分
2.对每一部分执行以下操作:
    将其中一部分作为当前测试集
    用剩余部分训练算法
    在当前测试集上测试算法
3.记录每次得分的平均得分
4.在上述过程中,每条数据只能在测试集中出现一次，以减少运气成分
'''
#使用sklearn提供交叉检验方法

scores = cross_val_score(estimator,X,Y,scoring = "accuracy")
average_accuracy = np.mean(scores) * 100
print("交叉验证的精确性是 {0:.1f}%".format(average_accuracy))



#设置参数
'''
近邻算法有多个参数,最重要的是选取多少个紧邻作为预测依据，scikit-learn管这个参数叫做n_neighbors
'''
avg_scores = []
all_scores = []
parameter_value = list(range(1,21))
for n_neighbors in parameter_value:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator,X,Y,scoring = "accuracy")
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

plt.plot(parameter_value,avg_scores,'-o')





