import numpy as np
import matplotlib.pylab as plt

'''
将gradient_descent.py和newton.py放在一个文件夹下
本程序是利用梯度下降法来寻找极小值点
定义了五个函数numerical_gradient(f, x)，Hessian(f, x)，function(x)，lr(f, x)，gradient_descent(f, init_x, step_num=100)
numerical_gradient(f,x)是用数值算法计算函数f在x上的梯度
Hessian（f，x）是计算函数f的hessian矩阵
funtion（x）是给出要求极小值的函数表达式
lr（f，x）是优化f在x时的学习率的值
gradient_descent(f, init_x, step_num=100)是给定初始值init_x时f下降step_num次后返回最后的x值和历次的x值
'''
def numerical_gradient(f, x):
    h = 1e-4#数值微分间隔
    grad = np.zeros_like(x)
    for dx in range(x.size):
        first_val = x[dx]
        x[dx] = first_val + h
        fxh1 = f(x)  # f(x+h)

        x[dx] = first_val - h
        fxh2 = f(x)  # f(x-h)
        grad[dx] = (fxh1 - fxh2) / (2 * h)

        x[dx] = first_val  # 还原值

    return grad


def Hessian(f, x):
    h = 1e-4
    Hess = np.zeros((len(x), len(x)))
    for dx in range(x.size):
        for dy in range(x.size):
            first_value = x[dy]
            x[dy] = first_value + h
            grad = numerical_gradient(f, x)
            dfh1 = grad[dx]

            x[dy] = first_value - h
            grad = numerical_gradient(f, x)
            dfh2 = grad[dx]
            Hess[dx, dy] = (dfh1 - dfh2) / (2 * h)

            x[dy] = first_value

    return Hess


def function(x):
    return x[0] ** 3 + x[1] ** 3 - 3 * x[0] * x[1]


def lr(f, x):
    grad = numerical_gradient(f, x)
    Hess = Hessian(f, x)
    epsi = np.dot(grad, grad) / np.dot(np.dot(grad, Hess), grad.T)
    return epsi


def gradient_descent(f, init_x, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        learningrate = lr(f, x)
        x -= learningrate * grad

    return x, np.array(x_history)


if __name__ == '__main__':
    init_x = np.array([3.0, 4.0])

    step_num = 20
    x, x_history = gradient_descent(function, init_x, step_num=step_num)
    print(x_history[-1,:])#输出最后的值
#画图
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(x_history[:, 0], x_history[:, 1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X")
    plt.ylabel("Y")

    step = 0.01
    x = np.arange(-4, 4, step)
    y = np.arange(-4, 4, step)
    X, Y = np.meshgrid(x, y)
    Z = X ** 3 + Y ** 3 - 3 * X * Y
    contour = plt.contour(X, Y, Z, 30, colors='k')
    plt.clabel(contour, fontsize=10, colors='k')
    plt.show()
