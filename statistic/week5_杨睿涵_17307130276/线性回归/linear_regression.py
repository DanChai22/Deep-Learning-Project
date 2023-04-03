import numpy as np
import matplotlib.pyplot as plt

# 输入训练集，测试集
train_x = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
train_y = [2.02139964787165, 2.22223769333024, 2.15337254906669, 1.98376882784720, 2.06261578185682, 1.88276651324467,
           1.95190334364623, 2.15862649836420, 2.45306063510008, 3.12490348700367]
test_x = [2, 2.2, 2.4, 2.6, 2.8, 3]
test_y = [3.91000499326572, 5.13943138623982, 6.65775784556462, 8.61502147852598, 11.0216461015868, 14.1233297112644]
plt.figure()
plt.title('Train-set\'s points')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(train_x, train_y, 'bo', color='#FDC453', label='True points')
plt.legend()
plt.show()
# 一次多项式拟合y=w0+w1x
# 构造一列全为1的列向量便于求[w0,w1]
constant1 = np.array([1] * len(train_x))
X_T = np.matrix((constant1, np.array(train_x)))
X = np.transpose(X_T)
Y = np.transpose(np.matrix(train_y))
# 利用公式直接求解W,Y=X*W,W=(X'*X)^(-1)*X'Y
W = np.linalg.inv(X_T * X) * X_T * Y
Y_fit = X * W
# 在测试集上求得x_test的预测值
constant2 = np.array([1] * len(test_x))
X_test = np.transpose(np.matrix((constant2, np.array(test_x))))
Y_result = X_test * W
# 计算均方误差
Y_test = np.transpose(np.matrix(test_y))
error1 = np.sum(np.transpose(Y_result - Y_test) * (Y_result - Y_test)) / len(test_x)
print('一次多项式估计的均方误差为：%f' % (error1))
# 画训练集的拟合图
y1 = Y_result.getA()
y_result = [y1[i][0] for i in range(len(test_y))]
y2 = Y_fit.getA()
y_fit = [y2[i][0] for i in range(len(train_y))]
x4plot = np.arange(0, 2, 0.01)
y4plot = W.getA()[0][0] + W.getA()[1][0] * x4plot
plt.figure()
plt.title('one-order linear for train set')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
plt.xlabel('x', font1)
plt.ylabel('y', font1)
# 画出拟合的一阶多项式，并标出拟合值和真实值
plt.plot(x4plot, y4plot, 'g', color='#F17173', label='1-linear')
plt.plot(train_x, train_y, 'bo', color='#FFBDB9', lw=2, label='True')
plt.plot(train_x, y_fit, 'bo', color='#CDE9DA', lw=2, label='Fit')
plt.legend()
plt.show()

# 三次多项式拟合:a0+a1x+a2x^2+a3x^3
# 构造一列全为1的列向量便于求[w0,w1]
constant1 = np.array([1] * len(train_x))
x1 = np.array(train_x)
x2 = x1 * x1
x3 = x1 * x1 * x1
X_T = np.matrix((constant1, x1, x2, x3))
X = np.transpose(X_T)
Y = np.transpose(np.matrix(train_y))
# 利用公式直接求解W,Y=X*W,W=(X'*X)^(-1)*X'Y
W = np.linalg.inv(X_T * X) * X_T * Y
Y_fit = X * W
# 在测试集上求得x_test的预测值
constant2 = np.array([1] * len(test_x))
x_test1 = np.array(test_x)
x_test2 = x_test1 * x_test1
x_test3 = x_test2 * x_test1
X_test = np.transpose(np.matrix((constant2, x_test1, x_test2, x_test3)))
Y_result = X_test * W
# 计算均方误差
Y_test = np.transpose(np.matrix(test_y))
error3 = np.sum(np.transpose(Y_result - Y_test) * (Y_result - Y_test)) / len(test_x)
print('3次多项式估计的均方误差为：%f' % (error3))
# 画出拟合的三阶多项式，并标出拟合值和真实值
y1 = Y_result.getA()
y_result = [y1[i][0] for i in range(len(test_y))]
y2 = Y_fit.getA()
y_fit = [y2[i][0] for i in range(len(train_y))]
x4plot = np.arange(0, 2, 0.01)
y4plot = W.getA()[0][0] + W.getA()[1][0] * x4plot + W.getA()[2][0] * x4plot * x4plot + W.getA()[3][
    0] * x4plot * x4plot * x4plot
plt.figure()
plt.title('3-order linear for train set')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
plt.xlabel('x', font1)
plt.ylabel('y', font1)
plt.plot(x4plot, y4plot, 'g', color='#FD8B64', label='3-linear')
plt.plot(train_x, train_y, 'bo', color='#FEB78D', lw=2, label='True')
plt.plot(train_x, y_fit, 'bo', color='#FCD353', lw=2, label='Predict')
plt.legend()
plt.show()

# 五次多项式拟合:y=a0+a1x+a2x^2+a3x^3+a4x^4+a5x^5
# 构造一列全为1的列向量便于求[w0,w1]
constant1 = np.array([1] * len(train_x))
x1 = np.array(train_x)
x2 = x1 * x1
x3 = x2 * x1
x4 = x3 * x1
x5 = x4 * x1
X_T = np.matrix((constant1, x1, x2, x3, x4, x5))
X = np.transpose(X_T)
Y = np.transpose(np.matrix(train_y))
# 利用公式直接求解W,Y=X*W,W=(X'*X)^(-1)*X'Y
W = np.linalg.inv(X_T * X) * X_T * Y
Y_fit = X * W
# 在测试集上求得x_test的预测值
constant2 = np.array([1] * len(test_x))
x_test1 = np.array(test_x)
x_test2 = x_test1 * x_test1
x_test3 = x_test2 * x_test1
x_test4 = x_test3 * x_test1
x_test5 = x_test4 * x_test1
X_test = np.transpose(np.matrix((constant2, x_test1, x_test2, x_test3, x_test4, x_test5)))
Y_result = X_test * W
# 计算均方误差
Y_test = np.transpose(np.matrix(test_y))
error5 = np.sum(np.transpose(Y_result - Y_test) * (Y_result - Y_test)) / len(test_x)
print('5次多项式估计的均方误差为：%f' % (error5))
# 画出拟合的五阶多项式，并标出拟合值和真实值
y = Y_result.getA()
y_result = [y[i][0] for i in range(len(test_y))]
y2 = Y_fit.getA()
y_fit = [y2[i][0] for i in range(len(train_y))]
x4plot = np.arange(0, 2, 0.01)
y4plot = W.getA()[0][0] + W.getA()[1][0] * x4plot + W.getA()[2][0] * x4plot ** 2 + W.getA()[3][0] * x4plot ** 3 + \
         W.getA()[4][0] * x4plot ** 4 + W.getA()[5][0] * x4plot ** 5
plt.figure()
plt.title('5-order linear for train set')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
plt.xlabel('x', font1)
plt.ylabel('y', font1)
plt.plot(x4plot, y4plot, 'g', color='#FDC453', label='5-linear')
plt.plot(train_x, train_y, 'bo', color='#FE8D6F', lw=2, label='True')
plt.plot(train_x, y_fit, 'bo', color='#9ADBC5', lw=2, label='Predict')
plt.legend()
plt.show()

# 九次多项式拟合:y=a0+a1x+a2x^2+a3x^3+a4x^4+a5x^5+a6x^6+a7x^7+a8x^8+a9x^9
# 构造一列全为1的列向量便于求[w0,w1]
constant1 = np.array([1] * len(train_x))
x1 = np.array(train_x)
x2 = x1 * x1
x3 = x2 * x1
x4 = x3 * x1
x5 = x4 * x1
x6 = x5 * x1
x7 = x6 * x1
x8 = x7 * x1
x9 = x8 * x1
X_T = np.matrix((constant1, x1, x2, x3, x4, x5, x6, x7, x8, x9))
X = np.transpose(X_T)
Y = np.transpose(np.matrix(train_y))
# 利用公式直接求解W,Y=X*W,W=(X'*X)^(-1)*X'Y
W = np.linalg.inv(X_T * X) * X_T * Y
Y_fit = X * W
# 在测试集上求得x_test的预测值
constant2 = np.array([1] * len(test_x))
x_test1 = np.array(test_x)
x_test2 = x_test1 * x_test1
x_test3 = x_test2 * x_test1
x_test4 = x_test3 * x_test1
x_test5 = x_test4 * x_test1
x_test6 = x_test5 * x_test1
x_test7 = x_test6 * x_test1
x_test8 = x_test7 * x_test1
x_test9 = x_test8 * x_test1
X_test = np.transpose(
    np.matrix((constant2, x_test1, x_test2, x_test3, x_test4, x_test5, x_test6, x_test7, x_test8, x_test9)))
Y_result = X_test * W
# 计算均方误差
Y_test = np.transpose(np.matrix(test_y))
error9 = np.sum(np.transpose(Y_result - Y_test) * (Y_result - Y_test)) / len(test_x)
print('9次多项式估计的均方误差为：%f' % (error9))
# 画出拟合的9阶多项式，并标出拟合值和真实值
y = Y_result.getA()
y_result = [y[i][0] for i in range(len(test_y))]
y2 = Y_fit.getA()
y_fit = [y2[i][0] for i in range(len(train_y))]
x4plot = np.arange(0, 2, 0.01)
y4plot = W.getA()[0][0] + W.getA()[1][0] * x4plot + W.getA()[2][0] * x4plot ** 2 + W.getA()[3][0] * x4plot ** 3 + \
         W.getA()[4][0] * x4plot ** 4 + W.getA()[5][0] * x4plot ** 5 + W.getA()[6][0] * x4plot ** 6 + W.getA()[7][
             0] * x4plot ** 7 + W.getA()[8][0] * x4plot ** 8 + W.getA()[9][0] * x4plot ** 9
plt.figure()
plt.title('9-order linear for train set')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
plt.xlabel('x', font1)
plt.ylabel('y', font1)
plt.plot(x4plot, y4plot, 'g', label='9-linear')
plt.plot(train_x, train_y, 'bo', color='#E7D3EE', lw=2, label='True')
plt.plot(train_x, y_fit, 'bo', color='#D395E0', lw=2, label='Predict')
plt.legend()
plt.show()

# 九次多项式拟合+L2正则项
# 正则项超参数
# alpha=0
# alpha=10
# alpha=10000
# alpha=1e15
alpha = 1e18 + 1e8 + 50000  # 中等大小
# alpha=1e20#很大的超参数
# 构造一列全为1的列向量便于求[w0,w1]
constant1 = np.array([1] * len(train_x))
x1 = np.array(train_x)
x2 = x1 * x1
x3 = x2 * x1
x4 = x3 * x1
x5 = x4 * x1
x6 = x5 * x1
x7 = x6 * x1
x8 = x7 * x1
x9 = x8 * x1
X_T = np.matrix((constant1, x1, x2, x3, x4, x5, x6, x7, x8, x9))
X = np.transpose(X_T)
Y = np.transpose(np.matrix(train_y))
# 利用公式直接求解W,Y=X*W,W=(X'*X)^(-1)*X'Y
# 当选取超参数过大时矩阵可能不可逆
try:
    W = np.linalg.inv(X_T * X + alpha * np.matrix(np.ones(10))) * X_T * Y
except:
    print('不存在逆矩阵')
    W = np.linalg.pinv(X_T * X + alpha * np.matrix(np.ones(10))) * X_T * Y
Y_fit = X * W
# 在测试集上求得x_test的预测值
constant2 = np.array([1] * len(test_x))
x_test1 = np.array(test_x)
x_test2 = x_test1 * x_test1
x_test3 = x_test2 * x_test1
x_test4 = x_test3 * x_test1
x_test5 = x_test4 * x_test1
x_test6 = x_test5 * x_test1
x_test7 = x_test6 * x_test1
x_test8 = x_test7 * x_test1
x_test9 = x_test8 * x_test1
X_test = np.transpose(
    np.matrix((constant2, x_test1, x_test2, x_test3, x_test4, x_test5, x_test6, x_test7, x_test8, x_test9)))
Y_result = X_test * W
# 计算均方误差
Y_test = np.transpose(np.matrix(test_y))
error9L2 = np.sum(np.transpose(Y_result - Y_test) * (Y_result - Y_test)) / len(test_x)
print('9次多项式+L2正则项后估计的均方误差为：%f' % (error9L2))
# 画出拟合的9阶多项式，并标出拟合值和真实值
y = Y_result.getA()
y_result = [y[i][0] for i in range(len(test_y))]
y2 = Y_fit.getA()
y_fit = [y2[i][0] for i in range(len(train_y))]
x4plot = np.arange(0, 2, 0.01)
y4plot = W.getA()[0][0] + W.getA()[1][0] * x4plot + W.getA()[2][0] * x4plot ** 2 + W.getA()[3][0] * x4plot ** 3 + \
         W.getA()[4][0] * x4plot ** 4 + W.getA()[5][0] * x4plot ** 5 + W.getA()[6][0] * x4plot ** 6 + W.getA()[7][
             0] * x4plot ** 7 + W.getA()[8][0] * x4plot ** 8 + W.getA()[9][0] * x4plot ** 9
plt.figure()
plt.title('9-order linear for train set(L2)')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
plt.xlabel('x', font1)
plt.ylabel('y', font1)
plt.plot(x4plot, y4plot, 'g', color='#71BBEE', label='9-linear(L2)')
plt.plot(train_x, train_y, 'bo', color='#EE84A8', lw=2, label='True')
plt.plot(train_x, y_fit, 'bo', color='#A3CEB8', lw=2, label='Predict')
plt.legend()
plt.show()
