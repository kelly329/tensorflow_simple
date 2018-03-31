# -*-coding:utf8-*-#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
每一张图片包含28像素X28像素 28*28=784 数组展开成一个向量 
"""
x = tf.placeholder("float", [None, 784])# 占位符 输入任意数量的mnist
# 权重
W = tf.Variable(tf.zeros([784,10]))

# 偏置
b = tf.Variable(tf.zeros([10]))

"""
实现回归模型
"""
y = tf.nn.softmax(tf.matmul(x,W) + b)

"""
训练模型
"""
y_ = tf.placeholder("float", [None,10]) #添加一个新的占位符用于输入正确值

# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#返向传播
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

"""
评估模型
"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))