import numpy as np
import matplotlib.pylab as plt
from gradient_descent import numerical_gradient
from gradient_descent import Hessian

'''
本程序是利用牛顿法法来寻找极小值点
从gradient_descent.py中导入了求梯度和Hessian矩阵的函数
并定义了newton_descent函数来衡量牛顿法的过程
'''
def newton_descent(f, init_x, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        hess = Hessian(f, x)
        grad = numerical_gradient(f, x)
        dx = np.dot(np.linalg.inv(hess), grad)

        x -= dx

    return x, np.array(x_history)


def function(x):
    return x[0] ** 3 + x[1] ** 3 - 3 * x[0] * x[1]


if __name__ == '__main__':
    init_x = np.array([3.0, 4.0])

    step_num = 20
    x, x_history = newton_descent(function, init_x, step_num=step_num)
    print(x_history[-1, :])#输出最后的值

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
