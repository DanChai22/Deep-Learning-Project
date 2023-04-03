import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

# 数据预处理
num_category = 10
imgrows, imgcols = 28, 28
input_shape = (imgrows, imgcols, 1)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], imgrows, imgcols, 1)
X_test = X_test.reshape(X_test.shape[0], imgrows, imgcols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

# 构建卷积网络
model = Sequential()

model.add(
    Conv2D(32, (2, 2), strides=(1, 1), activation='relu', input_shape=input_shape, padding='same', name="conv2d1"))
model.add(Dropout(rate=0.25))
model.add(Conv2D(32, (2, 2), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(32, (2, 2), strides=(1, 1), activation='relu', input_shape=input_shape, padding='same', name="conv2d2"))
model.add(Dropout(rate=0.25))
model.add(Conv2D(32, (2, 2), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(rate=0.25))

model.add(Dense(num_category, activation='softmax'))
# print(model.get_layer(name="conv2d1").output)
# print(model.get_weights())
# print(model.summary())

# 构建训练过程
# 定义损失函数、优化
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# 训练
train_history = model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=1)


def show_train_history(train_history, train):
    plt.plot(train_history.history[train])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train'], loc='upper left')
    plt.show()


print(train_history.history.keys())
show_train_history(train_history, 'accuracy')
# 预测
score = model.evaluate(X_test, y_test, verbose=1)
print(score[1])

model.save_weights("mnistCnnModel.h5")
print("Saved model to disk")
prediction = model.predict_classes(X_test)
# print(type(X_test))
# print(y_test.shape)
print(prediction[:20])
