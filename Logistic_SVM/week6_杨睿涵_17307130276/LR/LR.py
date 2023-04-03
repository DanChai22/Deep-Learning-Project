import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sb


# 为方便画图定义颜色
def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


# LR 召回率 精确率 准确率计算
def LR_correct(data, target, C_type):
    recall_all = np.zeros((1, len(C_type)))
    accuracy_all = np.zeros((1, len(C_type)))
    for i in range(len(C_type)):
        LR = LogisticRegression(C=C_type[i]).fit(data, target)
        predict_value = LR.predict(data)
        print('惩罚因子为:', C_type[i])
        print('真实标签为:')
        print(np.array(target))
        print('预测结果为:')
        print(predict_value)
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for k in range(len(predict_value)):
            if predict_value[k] == 1 and target[k] == 1:
                TP += 1
            elif predict_value[k] == 1 and target[k] == 0:
                FP += 1
            elif predict_value[k] == 0 and target[k] == 1:
                FN += 1
            elif predict_value[k] == 0 and target[k] == 0:
                TN += 1
        Precision = (TP + TN) / (TP + FN + FP + TN)
        print('准确率为%f' % (Precision))
        if (TP + FP) != 0:
            Accuracy = TP / (TP + FP)
            accuracy_all[0, i] = Accuracy
            print('精确度为%f' % (Accuracy))
        else:
            print('不存在1的正确分类')
            accuracy_all[0, i] = 0
        Recall = TP / (TP + FN)
        recall_all[0, i] = Recall
        print('召回率为%f' % (Recall))
        print('\n')
    fig2, axes = plt.subplots(1, 2)
    x = np.array(['0.1', '1', '10', '100'])
    y1 = recall_all[0, :]
    y2 = accuracy_all[0, :]
    axes[0].scatter(x, y1, marker='*')
    axes[0].plot(x, y1, color='#FCD353')
    axes[0].set_xlabel('C value', fontsize=12)
    axes[0].set_ylabel('Recall', fontsize=12)
    axes[1].scatter(x, y2)
    axes[1].plot(x, y2, color='#FCD353')
    axes[1].set_xlabel('C value', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    plt.title('LR accuracy && Recall')
    plt.show()


# LR可视化
def LR_plot(data, C_type):
    fig1, axes = plt.subplots(1, len(C_type))
    plt.subplots_adjust(wspace=0.35, hspace=0.3)
    # colorArr1 = ['#FCD353','#FD8B64','#FFBD89','#71BBEE','#FADEE1']
    # colorArr2 = ['#8474C7','#F6CF80','#FE8B84','r','#FADEE1']
    for i in range(len(C_type)):
        LR = LogisticRegression(C=C_type[i]).fit(data, target)
        h = 0.002
        x_min, x_max = data[:, 0].min() - 0.2, data[:, 0].max() + 0.2
        y_min, y_max = data[:, 1].min() - 0.2, data[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = LR.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axes[i].contour(xx, yy, Z, 1, colors='black', linewidths=0.85)
        axes[i].contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)
        axes[i].scatter(np.hstack((data[:6, 0], data[13:15, 0])), np.hstack((data[:6, 1], data[13:15, 1])), marker='x',
                        color='#FD8B64', s=15, lw=1.5)
        axes[i].scatter(np.hstack((data[6:13, 0], data[15:17, 0])), np.hstack((data[6:13, 1], data[15:17, 1])),
                        marker='o', color='#FCD353', s=13, lw=1.2)
        axes[i].set_xlabel(C_type[i], fontsize=10)
        if i == 0:
            axes[i].set_ylabel('LR', fontsize=10)
        axes[i].tick_params(axis="x", labelsize=7.5)
        axes[i].tick_params(axis="y", labelsize=7.5)
    plt.show()


def main():
    # 原始数据集
    data = np.array([
        [0.697, 0.460],
        [0.774, 0.376],
        [0.634, 0.264],
        [0.608, 0.318],
        [0.556, 0.215],
        [0.403, 0.237],
        [0.245, 0.057],
        [0.343, 0.099],
        [0.639, 0.161],
        [0.657, 0.198],
        [0.360, 0.370],
        [0.593, 0.042],
        [0.719, 0.103],
        [0.481, 0.149],
        [0.437, 0.211],
        [0.666, 0.091],
        [0.243, 0.267],
    ])
    target = [1] * 6 + [0] * 7 + [1] * 2 + [0] * 2
    # 可视化原始数据集
    plt.scatter(np.hstack((data[:6, 0], data[13:15, 0])), np.hstack((data[:6, 1], data[13:15, 1])), marker='o',
                color=randomcolor(), s=50, lw=2, label='label_1')
    plt.scatter(np.hstack((data[6:13, 0], data[15:17, 0])), np.hstack((data[6:13, 1], data[15:17, 1])), marker='x',
                color=randomcolor(), s=50, lw=2, label='label_0')
    plt.title('Original data')
    plt.legend()
    plt.show()
    return data, target


if __name__ == "__main__":
    data, target = main()
    C_type = [0.1, 1, 10, 100]
    LR_plot(data, C_type)
    LR_correct(data, target, C_type)
