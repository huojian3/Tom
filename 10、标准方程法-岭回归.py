
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


# In[3]:


# 读入数据
data = genfromtxt(r"G:\work\python\jupyter_notebook_work\机器学习\回归\longley.csv", delimiter = ',')
print(data)


# In[5]:


# 切分数据
x_data = data[1:, 2:]
y_data = data[1:, 1, np.newaxis]
print(x_data)
print(y_data)


# In[6]:


print(np.mat(x_data).shape)
print(np.mat(y_data).shape)
# 给样本添加偏置项
X_data = np.concatenate((np.ones((16, 1)), x_data), axis = 1)
print(X_data.shape)


# In[10]:


print(X_data[:3])


# In[14]:


# 岭回归标准方程法求解回归参数
def weights(xArr, yArr, lam = 0.2):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat   # 矩阵乘法
    rxTx = xTx + np.eye(xMat.shape[1]) * lam
    # 计算矩阵的值，如果值为0，说明该矩阵没有逆矩阵
    if np.linalg.det(rxTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    # xTx.I 为xTx的逆矩阵
    ws = rxTx.I * xMat.T * yMat
    return ws


# In[15]:


ws = weights(X_data, y_data)
print(ws)


# In[16]:


# 计算预测值
np.mat(X_data) * np.mat(ws)

