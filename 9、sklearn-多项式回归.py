
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression
# PolynomialFeatrues 使用多项式的方法来进行特征的构造
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


# 载入数据
data = genfromtxt(r"G:\work\python\jupyter_notebook_work\机器学习\回归\job.csv", delimiter = ',')
print(data)


# In[3]:


x_data = data[1:,1]
y_data = data[1:,2]
plt.scatter(x_data, y_data)
plt.show()


# In[4]:


x_data = data[1:, 1, np.newaxis]
y_data = data[1:, 2, np.newaxis]
# 创建并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)


# In[9]:


# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()


# In[42]:


# 定义多项式回归，degree的值可以调节多项式的特征（偏置值）
poly_reg = PolynomialFeatures(degree = 5)
# 特征处理
x_poly = poly_reg.fit_transform(x_data)
# 定义回归模型
lin_reg = LinearRegression()
# 训练模型
lin_reg.fit(x_poly, y_data)


# In[43]:


x_poly


# In[51]:


# 画图
# plt.plot(x_data, y_data, 'b')
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, lin_reg.predict(poly_reg.fit_transform(x_data)), c = 'r')
# 生成100个点的示例：(曲线更加平滑)
# x_test = np.linspace(1,10,100)
# x_test = x_test[:, np.newaxis]
# plt.plot(x_test, lin_reg.predict(poly_reg.fit_transform(x_test)), c = 'r')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

