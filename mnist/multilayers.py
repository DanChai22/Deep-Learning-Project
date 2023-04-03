import numpy as np
import keras
from keras.datasets import mnist
from collections import OrderedDict
import matplotlib.pylab as plt


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class MultiNet:
    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.params = {}
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            self.params['W' + str(idx)] = weight_init_std * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

        # 生成层
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 2):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Relu' + str(idx)] = Relu()

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):

        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads


def main():
    # 载入数据
    num_category = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    y_train = keras.utils.to_categorical(y_train, num_category)
    y_test = keras.utils.to_categorical(y_test, num_category)
    hidden_size_list = [300,300,300,300,300,300]
    keys = []
    for idx in range(1, len(hidden_size_list) + 2):
        keys.append('W' + str(idx))
        keys.append('b' + str(idx))

    network = MultiNet(input_size=X_train.shape[1], hidden_size_list=hidden_size_list, output_size=num_category)
    # 超参数
    batch_size = 100
    learning_rate = 0.01
    step_num = 1000
    training_size = X_train.shape[0]
    # 训练
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    for i in range(step_num):
        batch_mask = np.random.choice(training_size, batch_size)
        X_batch = X_train[batch_mask]
        y_batch = y_train[batch_mask]

        grad = network.gradient(X_batch, y_batch)

        for key in keys:
            network.params[key] -= learning_rate * grad[key]

        if i % 100 == 0:
            train_loss = network.loss(X_train, y_train)
            test_loss = network.loss(X_test, y_test)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_acc = network.accuracy(X_train, y_train)
            test_acc = network.accuracy(X_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
    print(train_loss_list, test_loss_list)
    print(train_acc_list, test_acc_list)
    # key_value = []
    # for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
    #     key_value.append(network.params[key])
    # print(key_value)
    # 画图
    f = plt.figure(figsize=(18, 10))

    f.add_subplot(2, 2, 1)
    plt.plot(train_acc_list, c='r', linestyle='--', marker='o')
    plt.ylabel("train_accuracy", fontsize=20)

    f.add_subplot(2, 2, 2)
    plt.plot(test_acc_list, c='r', linestyle='--', marker='o')
    plt.ylabel("test_accuracy", fontsize=20)

    f.add_subplot(2, 2, 3)
    plt.plot(train_loss_list, c='r', linestyle='--', marker='o')
    plt.ylabel("train_loss", fontsize=20)

    f.add_subplot(2, 2, 4)
    plt.plot(test_loss_list, c='r', linestyle='--', marker='o')
    plt.ylabel("test_loss", fontsize=20)

    plt.show()


if __name__ == '__main__':
    main()
