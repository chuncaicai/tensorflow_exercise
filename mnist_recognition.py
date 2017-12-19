# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf

#不同的session之间的数据和运算是相互独立的
sess = tf.InteractiveSession()
#None代表不限条数的输入
x = tf.placeholder(tf.float32, [None, 784])

#Variable在模型训练迭代中是持久化的，但是tensor一旦使用就会消失
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#softmax回归,其中tf.nn包含了大量神经网络组件
y = tf.nn.softmax(tf.matmul(x, W)+b)

#定义cross-entropy，出自信息熵
y_ = tf.placeholder(tf.float32, [None, 10])
#y_为真实的label
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
#tf.reduce_mean是用来对每个batch求平均值

#优化算法,确定学习速率以及优化的损失函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#全局参数初始化
tf.global_variables_initializer().run()

#对样本进行训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

#验证模型的准确率,tf.argmax为找出最大的概率的label
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#统计样本预测的accuracy,其中tf.cast是用来做类型转换的
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

