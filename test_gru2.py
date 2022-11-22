# _*__coding:utf-8 _*__
# @Time :2022/11/10 0010 22:08
# @Author :bay
# @File test_gru2.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# 整个流程还不太明白
import CustomGRU as GRU

state_size = 500
# CustomGRU.GRU2
gru2 = GRU.GRU2(state_size=state_size)
# print(gru2)
x = tf.keras.layers.Input((None, state_size))
# 这是啥意思 (None, None, 500)
# print(x.shape)
layer2 = tf.keras.layers.RNN(gru2)
# print(layer2)
y = layer2(x)
# (None, 500)
print(y.shape)
model2 = tf.keras.Model(inputs=x, outputs=y)
print(model2)
model2.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9))
X, Y = np.random.rand(1000, 20, state_size), np.random.rand(1000, state_size)
model2_results = model2.fit(X, Y, batch_size=50, epochs=50)
print(model2_results)
# plt.figure(figsize=(10, 10))
# y2 = model2_results.history['loss']
# x2 = range(1, len(y2) + 1)
# plt.plot(x2, y2, label='GRU2 type RNN')
# plt.xlim(0, 50)
# plt.legend()
# plt.show()