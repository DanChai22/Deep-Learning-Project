from sklearn.datasets import load_boston
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

'''
提取后六列作为数据
'''


def newdata():
    boston = load_boston()
    newdata = []
    for i in boston.data:
        newdata += [i[7:]]
    return newdata


def mpca(data, windows=10):
    newdata = data
    window_num = len(newdata) // windows  # 窗口个数
    count = 0  # 记录步骤
    pca = PCA(n_components=2)

    while count < window_num:
        window = newdata[count * windows:(count + 1) * windows]
        if count == 0:
            pca12 = pca.fit_transform(window)
            ev12 = pca.fit(window).explained_variance_ratio_  # 第一、二主成分信息占比
        else:
            pca12 = np.vstack((pca12, pca.fit_transform(window)))
            ev12 = np.vstack((ev12, pca.fit(window).explained_variance_ratio_))  # 第一、二主成分信息占比
        count += 1
    ev1 = [i[0] for i in ev12]
    ev1plus2 = [i[1] + i[0] for i in ev12]
    pca1 = [i[0] for i in pca12]
    pca2 = [i[1] for i in pca12]  # 第一、第二主成分

    # 画图
    fig, ax = plt.subplots()
    plt.xlim((0, window_num + 1))
    plt.ylim()
    plt.xlabel('window_num')
    plt.ylabel('rate')
    ax.plot([i for i in range(1, window_num + 1)], ev1, 'ro-', label='pca1')
    ax.plot([i for i in range(1, window_num + 1)], ev1plus2, 'bo-', label='pca1+pca2')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    data = newdata()
    mpca(data)
