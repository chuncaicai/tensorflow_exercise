# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST/", one_hot=True)
sess = tf.InteractiveSession()

#数据初始化
in_units = 784                     #输入节点数
h1_units = 300                     #隐含层输出节点数
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units],stddev=0.1))      #隐含层权重初始化为截断正态分布，标准差为0.1
#需要给正态分布加一点噪声打破完全对称并且避免0梯度
b1 = tf.Variable(tf.zeros(h1_units))
#某些模型还需要给偏置一些小的非0值来避免死亡神经元
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

#定义输入
x = tf.placeholder(tf.float32, [None, in_units])
#定义保留节点的概率
keep_prob = tf.placeholder(tf.float32)

#定义模型结构
hidden1 = tf.nn.relu(tf.matmul(x, W1)+b1)
#训练时keep_prob小于1，但是预测时=1
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2)+b2)

#定义输出
y_ = tf.placeholder(tf.float32, [None, 10])
#定义损失函数——交叉信息熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),
                                              reduction_indices=[1]))
#定义自适应优化器
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

#训练
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

#对模型进行准确率评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
