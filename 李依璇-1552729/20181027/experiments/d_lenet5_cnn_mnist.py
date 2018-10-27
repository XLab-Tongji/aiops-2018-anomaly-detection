import warnings
warnings.filterwarnings('ignore')

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r"E:\GitHub\Tensorflow-Tutorial-master\example-notebook\MNIST_data",one_hot=True)

print(mnist.test.labels.shape)
print(mnist.train.labels.shape)

import tensorflow as tf
sess=tf.InteractiveSession()

import numpy as np

def weight_variable(shape):
	#正态分布初始化权值
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	#本例使用relu激活函数，所以用一个小的正bias
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

# 卷积和池化API的第一个参数都是被操作的对象x

# 定义卷积层
def conv2d(x,W):
	# 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# 定义池化层
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

X_input = tf.placeholder(tf.float32,[None,784])
y_input = tf.placeholder(tf.float32,[None,10])


# 把X转为卷积所需要的形式
X = tf.reshape(X_input, [-1, 28, 28, 1]) #0-1位图

# 第一层卷积：5×5×1卷积核32个 [5，5，1，32]
W_conv1 = weight_variable([5,5,1,32]) # Kernal：5*5*1 32个
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X,W_conv1)+b_conv1) #卷积，偏置，relu，池化

#第一个pooling层
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积：5×5×32卷积核64个 [5，5，32，64]
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #卷积，偏置，relu，池化

#第二个pooling层
h_pool2 = max_pool_2x2(h_conv2)


# flatten
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

#fc1
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1) #线积，偏置，relu，池化

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)



cross_entropy = -tf.reduce_sum(y_input * tf.log(y_pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 3.预测准确结果统计
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_input, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 定义了变量必须要初始化，或者下面形式
sess.run(tf.global_variables_initializer())

import time


time0 = time.time()
# 训练
for i in range(5000):
    X_batch, y_batch = mnist.train.next_batch(batch_size=100)
    cost, acc,  _ = sess.run([cross_entropy, accuracy, train_step], feed_dict={X_input: X_batch, y_input: y_batch, keep_prob: 0.5})
    if (i+1) % 500 == 0:
        # 分 100 个batch 迭代
        test_acc = 0.0
        test_cost = 0.0
        N = 100
        for j in range(N):
            X_batch, y_batch = mnist.test.next_batch(batch_size=100)
            _cost, _acc = sess.run([cross_entropy, accuracy], feed_dict={X_input: X_batch, y_input: y_batch, keep_prob: 1.0})
            test_acc += _acc
            test_cost += _cost
        print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}; pass {}s".format(i+1, cost, acc, test_cost/N, test_acc/N, time.time() - time0))
        time0 = time.time()

