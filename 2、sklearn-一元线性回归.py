
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


# 载入数据
data = np.genfromtxt(r"G:\work\python\jupyter_notebook_work\机器学习\回归\data.csv",delimiter = ",")
x_data = data[:,0]
y_data = data[:,1]
plt.scatter(x_data, y_data)
plt.show()
print(x_data.shape)


# In[7]:


x_data = data[:, 0, np.newaxis]
print(x_data.shape)


# In[8]:


x_data = data[:, 0, np.newaxis]
y_data = data[:, 1, np.newaxis]
# 创建并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)


# In[9]:


# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()

