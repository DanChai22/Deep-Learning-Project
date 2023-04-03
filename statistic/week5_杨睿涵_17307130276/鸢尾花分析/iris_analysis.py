import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets

# 导入数据
iris = datasets.load_iris()  # iris是矩阵类型
iris_data = np.c_[iris.data, iris.target.T]
iris_dataframe = pd.DataFrame(iris_data)  # data是 data frame数据类型

# 因为转换后的的数据没有数据头标签，所以接下来添加标签
iris_dataframe.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']


# print(iris_dataframe.head)  # 测试代码，实际运行将被注释掉


def load_data():
    print(iris.keys())
    n_samples, n_features = iris.data.shape
    print((n_samples, n_features))
    print(iris.data[0])
    print(iris.target.shape)
    print(iris.target)
    print(iris.target_names)
    print("feature_names:", iris.feature_names)


# load_data()  # 试运行以获得数据集中的基本信息，在后续运行中将该行代码注释掉
'''运行结果为
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
(150, 4)
[5.1 3.5 1.4 0.2]
(150,)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
['setosa' 'versicolor' 'virginica']
feature_names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

由结果得知：
iris中有5个key值，其中：
iris.data 包含了四个特征值，例如[5.1, 3.5, 1.4, 0.2] 
iris.target为目标值 
iris.feature_names包含4个特征：
（Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、Petal.Length（花瓣长度）、Petal.Width（花瓣宽度））
特征值都为正浮点数，单位为厘米
'''


# 输出基本信息
def iris_information():
    print("Target␣names:␣{}".format(iris['target_names']))
    print("Feature␣names:␣{}".format(iris['feature_names']))
    print("Shape␣of␣data:␣{}".format(iris['data'].shape))
    return


# 对数据矩阵进行求均值、协方差矩阵以及相关系数矩阵
def iris_mean_cov_R():
    X = np.transpose(iris.data)  # 转置数据矩阵为（4,150）
    y = iris.target
    print("iris_Mean:\n{}".format(np.mean(X, axis=1)))  # axis=1 表示沿水平方向计算均值
    print("iris_Covariance␣Matrix:\n{}".format(np.cov(X)))
    print("iris_Correlation␣Matrix:\n{}".format(np.corrcoef(X)))
    return ()


# 通过Violinplot 和 Pointplot，分别从数据分布和斜率，观察各特征与品种之间的关系，通过Pairplot生成特征关系矩阵图
def violin_point_pair_plot():
    # 设置颜色主题
    antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']
    # 绘制  Violinplot
    f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    sns.despine(left=True)

    sns.violinplot(x='Species', y='SepalLengthCm', data=iris_dataframe, palette=antV, ax=axes[0, 0])
    sns.violinplot(x='Species', y='SepalWidthCm', data=iris_dataframe, palette=antV, ax=axes[0, 1])
    sns.violinplot(x='Species', y='PetalLengthCm', data=iris_dataframe, palette=antV, ax=axes[1, 0])
    sns.violinplot(x='Species', y='PetalWidthCm', data=iris_dataframe, palette=antV, ax=axes[1, 1])

    plt.show()

    # 绘制pointplot
    f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    sns.despine(left=True)

    sns.pointplot(x='Species', y='SepalLengthCm', data=iris_dataframe, color=antV[0], ax=axes[0, 0])
    sns.pointplot(x='Species', y='SepalWidthCm', data=iris_dataframe, color=antV[0], ax=axes[0, 1])
    sns.pointplot(x='Species', y='PetalLengthCm', data=iris_dataframe, color=antV[0], ax=axes[1, 0])
    sns.pointplot(x='Species', y='PetalWidthCm', data=iris_dataframe, color=antV[0], ax=axes[1, 1])

    plt.show()
    sns.pairplot(data=iris_dataframe, hue='Species')

    plt.show()

    return


# 最后，通过热图找出数据集中不同特征之间的相关性，高正值或负值表明特征具有高度相关性
def heatmapplot():
    X = np.transpose(iris.data)
    plt.subplots(figsize=(9, 9))
    ax = sns.heatmap(np.corrcoef(X), annot=True, vmax=1, square=True, cmap="Blues")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title("Correlation heat map of the four features")
    plt.show()

    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    fig = sns.heatmap(iris_dataframe.corr(), annot=True, cmap='RdBu_r',
                      linewidths=1, linecolor='k', square=True, mask=False,
                      vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"}, cbar=True)
    bottom, top = fig.get_ylim()
    fig.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()


def main():
    iris_information()
    iris_mean_cov_R()
    violin_point_pair_plot()
    heatmapplot()


if __name__ == '__main__':
    main()
