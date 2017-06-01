# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:38:26 2017

@author: Tywin
"""

#单变量线性回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression#sklearn线性回归

x = [[5],[7],[6],[11],[16],[20]]
y = [[1],[2],[3],[4],[7],[9]]

plt.title('披萨价格')
plt.xlabel('直径(英寸)')
plt.ylabel('价格')
plt.scatter(x,y)  #绘制散点图
'''
在 scikit-learn 里面，所有的估计器都带有 fit() 和 predict() 方法。 
fit() 用来分析模型参数，predict() 是通过 fit() 算出的模型参数构成的模型
，对解释变量进行预测获得的值。
'''
model = LinearRegression()
model.fit(x,y)
print('预测30英寸披萨价格 %.2f'%model.predict([30])[0])
y1 = model.predict([30])[0]
plt.scatter(30,y1)
x2 = [[3],[12],[5],[10],[30]]
y2 = model.predict(x2)
plt.plot(x2,y2)