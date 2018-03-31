# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义添加层的函数
def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 预测的值
    Wx_puls_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_puls_b
    else:
        outputs = activation_function(Wx_puls_b)
    return outputs

# x_data 300行 加入一个新的纬度 300*1
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 定义占位符
# xs None 表示无论给多少个sample都可以
xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])

# 隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

# 0.1 学习率
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# 显示真实数据 #
# 图片框
fig = plt.figure()
# 连续画图
ax = fig.add_subplot(1, 1, 1)
# 点的形式
ax.scatter(x_data, y_data)
# show以后不暂停 连续plt
plt.ion()
plt.show()

# 画线采用的方式先抹除再画线
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50:
        try:
            # 抹除上一条线
            ax.lines.remove(lines[0])
        except Exception:
            pass
        # 只要用到plaeholder的都要使用feed_dict
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        # 显示预测数据
        prediction_value = sess.run(prediction, feed_dict={xs:x_data, ys:y_data})
        # 使用曲线的形式 线宽为5 红色
        lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
        # 间隔0.1s
        plt.pause(0.1)


# 使用plt.ion后图像最后不能保留，把plt.show()放在最后（不在循环体中）,并在plt.show()之前添加 plt.ioff()



