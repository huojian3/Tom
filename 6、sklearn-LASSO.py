
# coding: utf-8

# In[1]:


import numpy as np
from numpy import genfromtxt
from sklearn import linear_model


# In[4]:


# 读入数据
data = genfromtxt(r"G:\work\python\jupyter_notebook_work\机器学习\回归\longley.csv", delimiter = ',')
print(data)


# In[5]:


# 切分数据
x_data = data[1:, 2:]
y_data = data[1:, 1]
print(x_data)
print(y_data)


# In[9]:


# 创建模型
model = linear_model.LassoCV()
model.fit(x_data, y_data)

# lasso系数
print(model.alpha_)
# 相关系数
print(model.coef_)
# lasso会使得某几个系数为0，这几个系数存在多重共线性。


# In[10]:


# 对某个数据做预测
model.predict(x_data[-2, np.newaxis])

