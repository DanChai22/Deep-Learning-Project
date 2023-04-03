import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

'''
提取后六列作为数据
'''


def newdata():
    boston = load_boston()
    newdata = []
    for i in boston.data:
        newdata += [i[7:]]
    return newdata


'''
数据标准化
'''


def normalization(X):
    Xmean = np.mean(X, 0)
    return (X - Xmean)


'''
计算数组协方差矩阵以及特征值和特征向量
'''


def eigen(X, k=2):
    CovX = np.cov(X.T)
    m, n = np.shape(X)
    evalue, evector = np.linalg.eig(CovX)
    index = np.argsort(-evalue)
    sumev = sum(evalue)
    aevalue = evalue / sumev  # 占比
    FData = []
    if k > n:
        print("k must be lower than n")
    else:
        selectVec = np.matrix(evector.T[index[:k]])
        selectaev = aevalue[index[:k]]
    return selectaev, selectVec


"""
參数：
    - data：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
    - k：表示取前k个特征值相应的特征向量
    -windows：时间窗口的中的样本数
    -printpca：是否按时间窗口打印pca
    -printpic：是否打印第一、第二主成分信息占比曲线
返回值：
    -result：将每一时间窗口所求得pca集中返回
"""


def pca(data, k=2, windows=10, printpca='False', printpic='True'):
    data_nom = normalization(data)
    m, n = np.shape(data_nom)
    windows_num = m // windows  # 设定窗口个数
    count = 0  # 记录步骤
    ev12 = []

    while count < windows_num:
        window = data_nom[count * windows:(count + 1) * windows]
        selectaev, selectVec = eigen(window, k)
        ev12 += [selectaev]
        fData = window * selectVec.T
        if count == 0:
            result = fData
        else:
            result = np.vstack((result, fData))
        if printpca == 'True':
            print('step:', count + 1)
            print(fData)
        elif printpca == 'False':
            print('', end='')
        else:
            print('Please input True or False.')
        count += 1
    ev1 = [i[0].real for i in ev12]
    ev1plus2 = [(i[1] + i[0]).real for i in ev12]

    # 画图
    fig, ax = plt.subplots()
    plt.xlim((0, windows_num + 1))
    plt.ylim()
    plt.xlabel('windows_num')
    plt.ylabel('rate')
    ax.plot([i for i in range(1, windows_num + 1)], ev1, 'ro-', label='pca1')
    ax.plot([i for i in range(1, windows_num + 1)], ev1plus2, 'bo-', label='pca1+pca2')
    ax.legend()
    plt.show()
    return result


if __name__ == "__main__":
    data = newdata()
    pca(data)
