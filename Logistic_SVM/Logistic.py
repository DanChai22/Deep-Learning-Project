import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradAscent(data, label):
    dataMatrix = np.mat(data)  # 转化为矩阵
    labelMat = np.mat(label).transpose()  # 转化为矩阵
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    Max = 600  # 迭代次数
    weight = np.ones((n, 1))
    weights_list = list()
    for k in range(Max):
        h = sigmoid(dataMatrix * weight)
        error = (labelMat - h)  # 代价函数
        weight = weight + alpha * dataMatrix.transpose() * error  # 优化参数
        weights_list.append(weight)
    return weight, weights_list


def logistic(data, label):
    weight, weights_list = gradAscent(data, label)
    y_test = sigmoid(data * weight)

    # 计算精确率和召回率
    num1 = num2 = num3 = 0
    for i in range(np.shape(data)[0]):
        if y_test[i] > 0.5:
            y_test[i] = 1
        else:
            y_test[i] = 0
        if (y_test[i] == label[i]) & (y_test[i] == 1):
            num1 += 1
        if y_test[i] == 1:
            num2 += 1
        if label[i] == 1:
            num3 += 1
    precision = num1 / num2
    recall = num1 / num3

    # 对点分类
    datax1 = []
    datax2 = []
    for i in data:
        datax1 += [i[0]]
        datax2 += [i[1]]
    color = []
    for i in label:
        if i == 0:
            color += ['r']
        if i == 1:
            color += ['k']

    # 画图
    plt.figure(1, figsize=(4, 3))
    plt.scatter(datax1, datax2, c=color, marker='o', s=50)

    plt.axis('tight')
    x_min = 0
    x_max = 1
    y_min = -0.5
    y_max = 1.0

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    predict = sigmoid(np.c_[XX.ravel(), YY.ravel()] * weight)
    predict = predict.reshape(XX.shape)
    plt.contour(XX, YY, predict, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.text(0, 0.8, 'precision:%.2f' % precision, fontsize=8)
    plt.text(0, 0.7, 'recall:%s' % recall, fontsize=8)
    plt.show()


if __name__ == "__main__":
    data = [[0.697, 0.46], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
            [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.36, 0.37], [0.593, 0.042],
            [0.719, 0.103], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267]]
    label = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    logistic(data, label)
