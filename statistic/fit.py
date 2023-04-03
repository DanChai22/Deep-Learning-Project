import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

x_train = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
y_train = [2.02139964787165, 2.22223769333024, 2.15337254906669, 1.98376882784720, 2.06261578185682, 1.88276651324467,
           1.95190334364623, 2.15862649836420, 2.45306063510008, 3.12490348700367]
x_test = [2, 2.2, 2.4, 2.6, 2.8, 3]
y_test = [3.91000499326572, 5.13943138623982, 6.65775784556462, 8.61502147852598, 11.0216461015868, 14.1233297112644]


def fitfunction(order):
    """

    :param order: 多项式的阶数
    :return: 多项式系数
    """
    coefficient = np.polyfit(x_train, y_train, order)
    return coefficient


def testerror(order):
    """

    :param order:    多项式的阶数
    :return:  测试误差
    """
    count = len(x_test)
    result = 0
    i = 0
    while i < count:
        result += (np.polyval(fitfunction(order), x_test[i]) - y_test[i]) ** 2
        i += 1
    return result / count


def xplot(select):
    """
    为函数作图提供x坐标
    :param select: 可选train or test
    :return: 输出10000个x 坐标
    """
    if select == 'train':
        return np.linspace(0, 1.8, 10000)
    if select == 'test':
        return np.linspace(2, 3, 10000)


def show(order=5):
    """

    :param order: 多项式的阶数
    :return: 没有返回，输出分别在训练集以及测试集上的函数图像
    """
    fig = plt.figure(figsize=(14, 7))
    plt.title('%d-degree polynomial fitting' % order)
    fig.add_subplot(1, 2, 1)
    plt.title('Fit the training set')
    plt.plot(xplot('train'), np.polyval(fitfunction(order), xplot('train')))
    plt.plot(x_train, y_train, 'b.', label='train data')
    plt.legend(loc='upper left')

    fig.add_subplot(1, 2, 2)
    plt.title('show the fit function on the test set')
    plt.plot(xplot('test'), np.polyval(fitfunction(order), xplot('test')))
    plt.plot(x_test, y_test, 'g^', label='test data')
    plt.legend(loc='lower left')
    plt.show()


def regularfunction(order=9, lam=0.0001, alpha=0.00665):
    """

    :param order:多项式的阶数
    :param lam:  ridge回归的常数系数
    :param alpha:  lasso回归的常熟系数
    :return: 没有返回，输出两种回归方法分别在训练集以及测试集上的函数图像 以及分别的测试误差
    """
    fig = plt.figure(figsize=(9, 9))
    plt.xticks([])
    plt.yticks([])


    poly = PolynomialFeatures(order)
    x_trainpoly = poly.fit_transform(np.array(x_train).reshape(10, 1))
    trainexample = np.array(np.linspace(0, 1.8, 10000)).reshape(10000, 1)
    testexample = np.array(np.linspace(2, 3, 10000)).reshape(10000, 1)

    deflasso = Lasso(alpha=alpha, max_iter=30000)
    deflasso.fit(x_trainpoly, y_train)
    y_pretrainlasso = deflasso.predict(poly.fit_transform(trainexample))
    y_pretestlasso = deflasso.predict(poly.fit_transform(testexample))

    defridge = Ridge(alpha=lam / 2, max_iter=30000)
    defridge.fit(x_trainpoly, y_train)
    ypretrainridge = defridge.predict(poly.fit_transform(trainexample))
    ypretestridge = defridge.predict(poly.fit_transform(testexample))

    fig.add_subplot(2, 2, 1)
    plt.title('use lasso to fit the training set')
    plt.plot(trainexample, y_pretrainlasso)
    plt.plot(x_train, y_train, 'g^', label='test data')

    fig.add_subplot(2, 2, 2)
    plt.title('lasso fit on the test set')
    plt.plot(testexample, y_pretestlasso)
    plt.plot(x_test, y_test, 'g^', label='test data')

    fig.add_subplot(2, 2, 3)
    plt.title('use ridge to fit the training set')
    plt.plot(trainexample, ypretrainridge)
    plt.plot(x_train, y_train, 'g^', label='test data')

    fig.add_subplot(2, 2, 4)
    plt.title('ridge fit on the test set')
    plt.plot(testexample, ypretestridge)
    plt.plot(x_test, y_test, 'g^', label='test data')
    plt.suptitle('9-degree polynomial fitting with the regularization')
    plt.show()

    count = len(x_test)
    lassoresult = 0
    ridgeresult = 0
    i = 0
    while i < count:
        lassoresult += deflasso.predict(poly.fit_transform((np.array(x_test[i]).reshape(1, 1))) - y_test[i]) ** 2
        ridgeresult += (defridge.predict(poly.fit_transform((np.array(x_test[i]).reshape(1, 1))) - y_test[i])) ** 2
        i += 1
    lassoresult = lassoresult / len(x_test)
    ridgeresult = ridgeresult / len(x_test)

    print('多项式阶数为9用L1正则化测试误差是：', float(lassoresult), '系数为', alpha)
    print('多项式阶数为9用L2正则化测试误差是：', float(ridgeresult), '系数为', lam)


if __name__ == '__main__':
    show(order=1)
    for order in [1, 3, 5, 9]:
        print('多项式阶数为%d测试误差是：' % order, testerror(order))
    regularfunction(order=9, lam=0.00665, alpha=0.00665)
