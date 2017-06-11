# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:15:53 2017
分类问题简单示例
@author: Tywin
"""

# 分类问题应用的目标，根据已知类别的数据集，经过训练得到一个分类模型，再用分类模型对类别未知的数据进行分类，比如垃圾邮件分类

# 准备数据集  Iris植物分类数据集  scikit-learn库内置

import numpy as np
from sklearn.datasets import load_iris
from collections import defaultdict
from operator import itemgetter
from sklearn.cross_validation import train_test_split

dataset = load_iris()
X = dataset.data
Y = dataset.target

attribute_means = x.mean(axis = 0) #每个特征的均值
X_d = np.array(X >=attribute_means,dtype = 'int')

def train_feature_value(X,y_true,feature_index,value):#根据预测数据的某项特征预测类别，并给出错误率
    class_counts = defaultdict(int)
    for sample, y in zip(X, y_true):
        if sample[feature_index] == value:
            class_counts[y] += 1 #遍历数据集中每一条数据，统计具有给定特征值的个体在各个类别中出现次数
    
    #对class_counts字典进行排序，找到最大值，就能找出具有给定特征值的个体在哪个类别中出现的次数最多
    sorted_class_counts = sorted(class_counts.items(),key=itemgetter(1))
    most_frequent_class = sorted_class_counts[0][0]
    
    #接着计算错误率，错误率为具有该特征值的个体在其他类别(除次数出现最多的类别之外的)中出现次数，它表示的是分类规则不适用的个体数量
    
    incorrect_predictions = [
            class_counts for class_value,class_count in class_counts.items() if class_value != most_frequent_class]
    error = sum(incorrect_predictions)
    
    return most_frequent_class,error

#获得总错误率与预测器

def train_on_feature(X,y_true,feature_index):
    values = set(X[:,feature_index])#返回所指列
    
    #创建预测器
    predictors = {}
    errors= [] #存储特征值的错误率
    
    # 遍历选定特征的每个不同的特征值，用train_feature_value函数找到每个特征最可能的类别，计算错误率，并将其分别保存到预测器
    #predictors和errors中
    
    for current_value in values:
        most_frequent_class,error = train_feature_value(X,y_true,feature_index,current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
        
    total_error = sum(errors)
    return predictors,total_error


Xd_train,Xd_test,Y_train,Y_test = train_test_split(X_d,Y,random_state = 14)#切分训练集和测试集

#接下来，计算所有特征值的目标类别(预测器)

all_predictors = {}
errors = {}

for feature_index in range(Xd_train.shape[1]):
    predictors,total_error = train_on_feature(Xd_train,Y_train,feature_index)
    all_predictors[feature_index] = predictors
    errors[feature_index] = total_error
    
#找出错误率最低的特征，作为分类的唯一规则
best_feature,best_error = sorted(errors.items,key = itemgetter(1))[0]

#对预测器进行排序，找到最佳特征值
# model 包含两个元素，用于分类的特征和预测器
model = {'feature':best_feature,'predictor':all_predictors[best_feature][0]}


def predict(X_test,model):
    variable = model['variable']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted

y_predicted = predict(X_test,model)



































    