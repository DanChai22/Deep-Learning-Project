from sklearn import svm
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
import matplotlib.pyplot  as plt
import joblib
import time


def visualization(no=0, set='train'):
    """
    :param no: 第几张数据
    :param set: 训练集还是测试集
    :return: 训练集或测试集任一一张数据、训练集和测试集各个数字出现的频数统计、训练集前100张数据
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if set == 'train':
        xp = X_train[no]
        yno = y_train[no]
    else:
        xp = X_test[no]
        yno = y_test[no]
    plt.imshow(xp.reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.title('the label of picture %d in %s set is %d.' % (no + 1, set, yno))
    plt.show()

    # 统计训练集和测试集个数字出现的频数
    yltr = []
    ylte = []
    xno = []
    for i in range(0, 10):
        yltr += [list(y_train).count(i)]
        ylte += [list(y_test).count(i)]
        xno += [i]
    if set == 'train':
        plt.bar(xno, yltr, tick_label=xno)
        plt.title('Training set quantity statistics')
        plt.show()
    else:
        plt.bar(xno, ylte, tick_label=xno)
        plt.title('Test set quantity statistics')
        plt.show()

    # 输出训练集前100张数据
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True)

    ax = ax.flatten()
    for i in range(100):
        a = y_train[i]
        img = X_train[y_train == a][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)
    # 模型训练以及存储，并可输出训练时间。模型已经训练结束，直接调用
    t1 = time.time()
    clf = clf.fit(X_train, y_train)
    t2 = time.time()
    trainefficiency = t2 - t1
    joblib.dump(clf, 'clf.model')
    print('训练时间：', trainefficiency)

    # 模型调用
    clf = joblib.load('clf.model')
    tt1 = time.time()
    predictions = [int(a) for a in clf.predict(X_test)]
    tt2 = time.time()
    print('accuracy:', accuracy_score(y_test, predictions))
    print('测试时间：', tt2 - tt1)


if __name__ == "__main__":
    main()
