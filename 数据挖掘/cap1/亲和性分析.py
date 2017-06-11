# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:54:04 2017
亲和性分析根据样本个体之间的相似度确定他们的亲疏，多应用于个性化广告投放，推荐内容，根据基因寻找亲缘关系的人
@author: Tywin
"""

# 目的‘如果顾客购买了商品X，那么他们可能愿意购买商品Y’
import numpy as np
dataset_filename = "affinity_dataset.txt"
x = np.loadtxt(dataset_filename)#  以此代表 面包，牛奶，奶酪，苹果，香蕉五种商品购买的记录  每个特征只有两个可能的值

# 判断交易中顾客是否买了苹果
num_apple_purchases = 0
for sample in x:
    if sample[3] == 1:
        num_apple_purchases += 1

print("{0}人买了苹果".format(num_apple_purchases))