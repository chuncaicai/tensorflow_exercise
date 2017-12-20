# -*- coding: utf-8 -*-
#建立完整的AlexNet网络，对batch的forward,backward的速度进行测试

from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

#定义用来显示网络每一层结构的函数，展示每一个tensor的尺寸
def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())

#定义AlexNet的网络结构
#定义inference函数,images为输入，训练所需参数为输出
def inference(images):
    parameters = []

    #定义第一个卷积层,name_scope可以将scope内生成的Variable自动命名为conv1/xxx
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                                 dtype=tf.float32,
                                                 stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

        #在卷积层后面添加LRN和最大池化层
   #     lrn1 = tf.nn.lrn(conv1, 4, bias= 1.0, alpha=0.001/9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')

        print_activations(pool1)

    #定义第二个卷积层
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                 dtype=tf.float32,
                                                 stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)

        # 在卷积层后面添加LRN和最大池化层
    #   lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool2')
        print_activations(pool2)

    # 定义第三个卷积层
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    # 定义第四个卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    # 定义第五个卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

        #接最大池化层
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool5')
        print_activations(pool5)

    return pool5, parameters

#全连接层对计算速度的影响很小，不考虑

#定义一个每轮时间的计算函数
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10           #跳过头几轮
    total_duration = 0.0             #总时间和平方
    total_duration_square = 0.0      #计算方差

    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i%10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i-num_steps_burn_in, duration))
            total_duration += duration
            total_duration_square += duration*duration

    mn = total_duration/num_batches                        #每轮迭代平均耗时
    vr = total_duration_square/num_batches-mn*mn
    sd = math.sqrt(vr)                                     #标准差
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' %
          (datetime.now(), info_string, num_batches, mn, sd))

#主函数
def run_benchmark():
    with tf.Graph().as_default():      #随机图片数据测试forward和backward的速度
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size,3],
                                              dtype=tf.float32,
                                              stddev=1e-1))

        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        #评测forward
        time_tensorflow_run(sess, pool5, "Forward")

        #评测Backward，设置优化目标loss和求梯度参数的过程，参数更新用时较少，不算在内
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Backward")

run_benchmark()

