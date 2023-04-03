import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# 均值可视化
def visualme(data):
    imean = []
    for i in range(3):
        for j in range(4):
            imean.append(data.values[i * 50:50 * i + 49, j].mean())
    for k in range(4):
        imean.append(data.values[:, k].mean())
    mean_array = np.array(imean).reshape((4, 4))
    # 画图
    target = ['setosa', 'versicolor', 'virginica', 'all']
    feature = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), dpi=100)
    x = feature
    for i in range(2):
        for j in range(2):
            y = mean_array[2 * i + j]
            axes[i, j].set_title(target[2 * i + j])
            axes[i, j].bar(x, y, color='darkblue')
    plt.suptitle('Average visualization')
    return mean_array


# 方差可视化
def visualva(data):
    ivariance = []
    for i in range(3):
        for j in range(4):
            ivariance.append(data.values[i * 50:50 * i + 49, j].var())
    for k in range(4):
        ivariance.append(data.values[:, k].var())
    varicane_array = np.array(ivariance).reshape((4, 4))
    # 画图
    target = ['setosa', 'versicolor', 'virginica', 'all']
    feature = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), dpi=100)
    x = feature
    for i in range(2):
        for j in range(2):
            y = varicane_array[2 * i + j]
            axes[i, j].set_title(target[2 * i + j])
            axes[i, j].bar(x, y, color='darkblue')
    plt.suptitle('variance visualization')
    return varicane_array


if __name__ == '__main__':
    iris_data = sns.load_dataset("iris")
    print(iris_data.describe())  # 查看数据的统计信息

    sns.set_style('dark')
    g = sns.pairplot(iris_data, hue='species', palette='husl')  # 查看总体数据

    data = iris_data
    print(visualme(data))
    print(visualva(data))
    plt.show()
