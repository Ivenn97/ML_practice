"""
线性回归知识：https://zhuanlan.zhihu.com/p/80887841

TensorFlow 实现
"""

# TensorFlow 2.0 + Linear Regression

import tensorflow as tf
import numpy as np

x = np.float32(np.random.rand(100, 1))

# y=a*x+b
y = np.dot(x, 0.8) + 0.2

a = tf.Variable(np.float32())
b = tf.Variable(np.float32())


def model(x):
    return a * x + b


def loss(predicted_y, desired_y):
    return tf.reduce_sum(tf.square(predicted_y - desired_y))


optimizer = tf.optimizers.Adam(0.1)

for step in range(0, 101):
    with tf.GradientTape() as t:
        outputs = model(x)
        current_loss = loss(outputs, y)
        grads = t.gradient(current_loss, [a, b])
        optimizer.apply_gradients(zip(grads, [a, b]))
    if step % 10 == 0:
        print("Step:%d, loss:%2.5f, weight:%2.5f, bias:%2.5f "
              % (step, current_loss.numpy(), a.numpy(), b.numpy()))
