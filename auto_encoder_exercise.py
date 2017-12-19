# -*- coding: utf-8 -*-
#去噪自编码器

import numpy as np
import sklearn.preprocessing as prep   #做数据预处理
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#权重初始化，xavier的作用为让权重满足0均值，方差为2/(fin_in+fin_out)
#fan_in为输入节点的数量
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

#定义去噪自编码类，n_input(输入变量数),n_hidden(隐含层节点数)，transfer_function(隐含层激活函数)
#optimizer优化器
#此代码中只有一个隐藏层
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32) #class内的scale
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        #输入
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        #隐藏层scale*tf.random为所增加的噪声
        print(self.weights['b1'].shape)
        print(tf.matmul(self.x+scale*tf.random_normal((n_input,)),self.weights['w1']).shape)
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x+scale*tf.random_normal((n_input,)),
            self.weights['w1']), self.weights['b1']))
        #建立层
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        #定义损失函数
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))     #平方误差
        #定义优化器
        self.optimizer = optimizer.minimize(self.cost)

        #建立session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    #定义参数初始化函数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                  self.n_input],
                                                 dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([ self.n_input],
                                                 dtype=tf.float32))
        return all_weights

    #计算损失函数以及进行一步训练
    def partial_fit(self, X):
        cost , opt = self.sess.run((self.cost, self.optimizer),
                                   feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    #只求损失函数而不进行训练的函数
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    #定义自编码隐含层输出结果(学习数据中的高阶特征)
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    #将高阶特征复原为原始数据
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    #定义重建函数，将transform与generate相结合
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                             self.scale: self.training_scale})

    #获取隐含层的权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    #获得隐含层的偏置系数
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


#测试AGN
mnist = input_data.read_data_sets('MNIST/', one_hot=True)

#对数据进行标准化处理的函数,让数据的均值为0，方差为1,（减去均值再除以标准差StandardScaler）
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

#定义一个获取随机block的函数，不放回抽样
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

#对数据进行标准化变换
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

#创建一个AGN自编码器的实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)

#开始训练过程
for epoch in range(training_epochs):
    avg_cost = 0   #平均损失
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size

    if epoch%display_step == 0:
        print('Epoch:', '%04d'%(epoch+1), 'cost=', "{:.9f}".format(avg_cost))

print("Total cost:"+str(autoencoder.calc_total_cost(X_test)))


