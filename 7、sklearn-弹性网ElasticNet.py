
# coding: utf-8

# In[1]:


import numpy as np
from numpy import genfromtxt
from sklearn import linear_model


# In[2]:


# 读入数据
data = genfromtxt(r"G:\work\python\jupyter_notebook_work\机器学习\回归\longley.csv", delimiter = ',')
print(data)


# In[4]:


# 切分数据
x_data = data[1:, 2:]
y_data = data[1:, 1]
print(x_data)
print(y_data)


# In[5]:


# 创建模型
model = linear_model.ElasticNetCV()
model.fit(x_data, y_data)

# 弹性网系数
print(model.alpha_)
# 相关系数
print(model.coef_)


# In[8]:


# 对选定数据做预测
model.predict(x_data[-3, np.newaxis])

