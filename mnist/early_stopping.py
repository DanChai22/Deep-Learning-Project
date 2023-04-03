import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import tensorflow as tf
import time

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
a = 600
model = Sequential()

model.add(Flatten(input_shape=input_shape))
model.add(Dense(a, activation='relu'))
model.add(Dense(a, activation='relu'))
model.add(Dense(a, activation='relu'))
model.add(Dense(a, activation='relu'))
model.add(Dense(a, activation='relu'))
model.add(Dense(a, activation='relu'))
model.add(Dense(a, activation='relu'))
model.add(Dense(a, activation='relu'))
model.add(Dense(a, activation='relu'))
model.add(Dense(a, activation='relu'))
model.add(Dense(num_category, activation='softmax'))
# print(model.get_weights())
# print(model.summary())

# 构建训练过程
# 定义损失函数、优化
keras.optimizers.Adam(lr=100, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=2, mode='auto', baseline=None,
                                       restore_best_weights=False)
t1 = time.time()
# 训练
train_history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=128, verbose=1,callbacks=es)

t2 = time.time()
print('time is :', t2 - t1)
epochs = len(train_history.history['loss'])
plt.plot(range(epochs), train_history.history['loss'], label='loss')
plt.plot(range(epochs), train_history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss in train set and validation set')
plt.title('node number:%d,dense:8' % a)
plt.legend()
plt.show()
