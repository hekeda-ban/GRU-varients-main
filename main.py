# _*__coding:utf-8 _*__
# @Time :2022/11/10 0010 17:23
# @Author :bay
# @File main.py.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import CustomGRU as GRU

state_size = 500

gru0, gru1, gru2, gru3 = GRU.CustomGRU(state_size), GRU.GRU1(state_size), GRU.GRU2(state_size), GRU.GRU3(state_size)

x = tf.keras.layers.Input((None, state_size))

layer0, layer1, layer2, layer3 = tf.keras.layers.RNN(gru0), tf.keras.layers.RNN(gru1), \
                                 tf.keras.layers.RNN(gru2), tf.keras.layers.RNN(gru3)

y0, y1, y2, y3 = layer0(x), layer1(x), layer2(x), layer3(x)

model0, model1, model2, model3 = tf.keras.Model(inputs=x, outputs=y0), tf.keras.Model(inputs=x, outputs=y1), \
                                 tf.keras.Model(inputs=x, outputs=y2), tf.keras.Model(inputs=x, outputs=y3)

native_tensorflow_gru = tf.keras.layers.GRU(state_size)
y4 = native_tensorflow_gru(x)
model4 = tf.keras.Model(inputs=x, outputs=y4)

model0.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9))
model1.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9))
model2.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9))
model3.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9))
model4.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9))
X, Y = np.random.rand(1000, 20, state_size), np.random.rand(1000, state_size)
model0_results = model0.fit(X, Y, batch_size=50, epochs=50)
model1_results = model1.fit(X, Y, batch_size=50, epochs=50)
model2_results = model2.fit(X, Y, batch_size=50, epochs=50)
model3_results = model3.fit(X, Y, batch_size=50, epochs=50)
model4_results = model4.fit(X, Y, batch_size=50, epochs=50)


plt.figure(figsize=(10, 10))
y4 = model4_results.history['loss']
x4 = range(1, len(y4) + 1)
plt.plot(x4, y4, label='Native TensorFlow GRU Implementation')
plt.xlim(0, 50)
#plt.ylim(1, 0)
y0 = model0_results.history['loss']
x0 = range(1, len(y0) + 1)
plt.plot(x0, y0, label='Classical GRU RNN')
plt.xlim(0, 50)
#plt.ylim(1, 0)
y1 = model1_results.history['loss']
x1 = range(1, len(y1) + 1)
plt.plot(x1, y1, label='GRU1 type RNN')
plt.xlim(0, 50)
#plt.ylim(1, 0)
y2 = model2_results.history['loss']
x2 = range(1, len(y2) + 1)
plt.plot(x2, y2, label='GRU2 type RNN')
plt.xlim(0, 50)
#plt.ylim(1, 0)
y3 = model3_results.history['loss']
x3 = range(1, len(y3) + 1)
plt.plot(x3, y3, label='GRU3 type RNN')
plt.xlim(0, 50)
#plt.ylim(1, 0)
plt.legend()
plt.show()