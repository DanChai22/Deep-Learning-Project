import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sb


# 为方便画图定义颜色
def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


# svm 召回率 精确率 准确率计算
def SVM_correct(data, target, kernel_type, C_type):
    recall_all = np.zeros((len(kernel_type), len(C_type)))
    accuracy_all = np.zeros((len(kernel_type), len(C_type)))
    for i in range(len(kernel_type)):
        for j in range(len(C_type)):
            svc_My = svm.SVC(kernel=kernel_type[i], C=C_type[j]).fit(data, target)
            predict_value = svc_My.predict(data)
            print('核函数为:', kernel_type[i])
            print('惩罚因子为:', C_type[j])
            print('真实标签为:')
            print(np.array(target))
            print('预测结果为:')
            print(predict_value)
            # 这里假设原本分类为1的样本为正样本
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
            print('准确率为%f' % Precision)
            if (TP + FP) != 0:
                Accuracy = TP / (TP + FP)
                accuracy_all[i][j] = Accuracy
                print('精确度为%f' % Accuracy)
            else:
                print('不存在1的正确分类')
                Accuracy = 0
                accuracy_all[i][j] = 0
            Recall = TP / (TP + FN)
            recall_all[i][j] = Recall
            print('召回率为%f' % (Recall))
            # F1为召回率与精确度的平衡点
            if (TP + FP) != 0:
                F1 = 2 * Recall * Accuracy / (Recall + Accuracy)
                print('F1为%f' % F1)
                print('\n')
    colorArr = ['#FCD353', '#FD8B64', '#FADEE1', '#B0D197']
    fig2, axes = plt.subplots(1, 2)
    for i in range(len(kernel_type)):
        x = np.array(['0.1', '1', '10', '100'])
        y1 = recall_all[i, :]
        y2 = accuracy_all[i, :]
        axes[0].scatter(x, y1, marker='*')
        axes[0].plot(x, y1, color=colorArr[i], label=kernel_type[i])
        axes[0].set_xlabel('C value', fontsize=12)
        axes[0].set_ylabel('Recall', fontsize=12)
        axes[1].scatter(x, y2)
        axes[1].plot(x, y2, color=colorArr[i], label=kernel_type[i])
        axes[1].set_xlabel('C value', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
    plt.title('SVM accuracy && Recall')
    plt.legend()
    plt.show()


# svm可视化
def SVM_plot(data, kernel_type, C_type):
    fig1, axes = plt.subplots(len(kernel_type), len(C_type))
    plt.subplots_adjust(wspace=0.35, hspace=0.3)
    # colorArr1 = ['#FCD353','#FD8B64','#FFBD89','#71BBEE','#FADEE1']
    # colorArr2 = ['#8474C7','#F6CF80','#FE8B84','r','#FADEE1']
    for i in range(len(kernel_type)):
        for j in range(len(C_type)):
            svc_My = svm.SVC(kernel=kernel_type[i], C=C_type[j]).fit(data, target)
            h = 0.002
            x_min, x_max = data[:, 0].min() - 0.2, data[:, 0].max() + 0.2
            y_min, y_max = data[:, 1].min() - 0.2, data[:, 1].max() + 0.2
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = svc_My.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            Z1 = svc_My.predict(np.c_[xx.ravel(), yy.ravel()])
            Z1 = Z1.reshape(xx.shape)
            # axes[i,j].contour(xx, yy,Z,1, colors='black',linewidths=0.85)
            axes[i, j].contourf(xx, yy, Z1, cmap=plt.cm.ocean, alpha=0.6)
            axes[i, j].contour(xx, yy, Z, colors=['black', 'black', 'black'], linestyles=['--', '-', '--'],
                               levels=[0, 0.5, 1], linewidths=0.85)
            axes[i, j].scatter(np.hstack((data[:6, 0], data[13:15, 0])), np.hstack((data[:6, 1], data[13:15, 1])),
                               marker='x', color='#FD8B64', s=15, lw=1.5)
            axes[i, j].scatter(np.hstack((data[6:13, 0], data[15:17, 0])), np.hstack((data[6:13, 1], data[15:17, 1])),
                               marker='o', color='#FCD353', s=12.5, lw=1.2)
            data_predict = svc_My.decision_function(data)
            # 画出错误的点与support vector
            for s in range(len(data_predict)):
                if abs(data_predict[s] - 1) <= 1e-3 or abs(data_predict[s]) <= 1e-3:
                    axes[i, j].scatter(data[s, 0], data[s, 1], marker='o', c='', edgecolors='black', s=15)
                if data_predict[s] < 0.5 and target[s] == 1:
                    axes[i, j].scatter(data[s, 0], data[s, 1], marker='+', c='#D35A7F', s=14)
                if data_predict[s] > 0.5 and target[s] == 0:
                    axes[i, j].scatter(data[s, 0], data[s, 1], marker='+', c='#D35A7F', s=14)

            if i == 3:
                axes[i, j].set_xlabel(C_type[j], fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(kernel_type[i], fontsize=10)
            axes[i, j].tick_params(axis="x", labelsize=7.5)
            axes[i, j].tick_params(axis="y", labelsize=7.5)

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
    kernel_type = ['linear', 'poly', 'rbf', 'sigmoid']
    C_type = [0.1, 1, 10, 100]
    SVM_plot(data, kernel_type, C_type)
    SVM_correct(data, target, kernel_type, C_type)
