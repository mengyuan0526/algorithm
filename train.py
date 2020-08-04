import numpy as np 
import matplotlib.pyplot as plt 
from Linear_regression import *

def show_data(x,y,w=None,b=None):
    plt.scatter(x,y,marker='.')
    if w is not None and b is not None:
        plt.scatter(x,w*x+b,c='red')
    plt.show()

#生成数据
np.random.seed(272)
data_size=100
x = np.random.uniform(low=1.0,high=10.0,size=data_size)
y = x*20 +10 +np.random.normal(0.0,10.0,data_size)

plt.scatter(x,y,marker='.')
plt.show()

#训练数集、测试数集
shuffled_index = np.random.permutation(data_size)
x = x[shuffled_index]
y = y[shuffled_index]
split_index = int(data_size*0.7)
x_train = x[:split_index]
y_train = y[:split_index]
x_test = x[split_index:]
y_test = x[split_index:]


plt.scatter(x_train,y_train,marker='.')
plt.show()

plt.scatter(x_test,y_test,marker='.')
plt.show()

#拟合Linear_regression模型
regr = LinerRegression(learning_rate=0.01,max_iter=10,seed=314)
regr.fit(x_train,y_train)
print('cost:\t{:.3}'.format(regr.loss()))
print('w:\t{:.3}'.format(regr.w))
print('b:\t{:.3}'.format(regr.b))
show_data(x,y,regr.w,regr.b)

##验证
from sklearn import linear_model
#创建线性回归训练器
x_train1 = np.matrix(x_train)
y_train1 = np.matrix(y_train)
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(x_train1,y_train1)
linear_regressor.predict(x_train1)
linear_regressor.intercept_
coef_fit = linear_regressor.coef_




