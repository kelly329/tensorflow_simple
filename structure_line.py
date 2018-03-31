# -- coding: utf-8 --
import tensorflow as tf
# numpy 数据矩阵化
import numpy as np

# 创造数据  生成100个随机数列
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 创建tensorflow structure start #
# weights 可能为矩阵  定义随机数列生成
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data +biases
# tf.reduce_mean 求平均值  square 是平方的意思
loss = tf.reduce_mean(tf.square(y-y_data))
# 0.5 learning rate 学习效率 学习率决定了参数移动到最优值的速度快慢。
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
# 创建tensorflow structure end #

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0 :
        print(step, sess.run(Weights), sess.run(biases))

