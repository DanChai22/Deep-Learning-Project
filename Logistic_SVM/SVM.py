import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def SVM(data, label, C=10):
    X = np.array(data)
    Y = label
    fignum = 1

    for kernel in ('linear', 'poly', 'rbf'):
        clf = svm.SVC(kernel=kernel, C=C,degree=2)
        clf.fit(X, Y)

        # 画图
        plt.figure(fignum, figsize=(4, 3))
        plt.clf()

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                    edgecolors='k')

        plt.axis('tight')
        x_min = 0
        x_max = 1
        y_min = -0.5
        y_max = 1.5

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # 计算召回率和精确率
        recall = recall_score(Y, clf.predict(data), average='binary')
        precision = precision_score(Y, clf.predict(data), average='binary')
        plt.text(0, 1.42, 'precision:%s' % precision, fontsize=8)
        plt.text(0, 1.28, 'recall:%s' % recall, fontsize=8)

        fignum = fignum + 1
    plt.show()


if __name__ == "__main__":
    data = [[0.697, 0.46], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
            [0.245, 0.057],
            [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.36, 0.37], [0.593, 0.042], [0.719, 0.103],
            [0.481, 0.149],
            [0.437, 0.211], [0.666, 0.091], [0.243, 0.267]]
    Y = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    SVM(data, Y, C=10)
