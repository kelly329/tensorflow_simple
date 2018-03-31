# -*-coding:utf8-*-#
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


"""
输出准确度
"""
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result
"""
定义权重
"""
def weight_variable(shape):
    # 产生随机变量
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

"""
定义偏置
开始有值 不为0
"""
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
"""
定义卷积神经网络层
x输入 w权重 strides步长 四个长度的列表
"""
def conv2d(x,W):
    # strides = [1,x移动跨度，y移动跨度,1]
    # SAME 图片大小不会改变 高度发生变化  valid图片大小发生
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

"""
定义池化层
使用最大池化函数
ksize 池化窗口的大小，卷积核的大小，四位向量
"""
def max_pool_2x2(x):
    # 2个像素移动一下
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 定义占位符并入到神经网络
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,784],name='x_input') # 28*28
    ys = tf.placeholder(tf.float32,[None,10], name='y_input')
keep_prob = tf.placeholder(tf.float32)
# 1图片黑白的 3：rgb
x_image= tf.reshape(xs,[-1,28,28,1])
# print(x_image.shape) #[n_samples,28,28,1]

"""
卷积第一层
"""
# 5*5的过滤器 1图像的高度 32输入的高度
with tf.name_scope('layer1'):
    with tf.name_scope('weights1'):
        W_conv1 = weight_variable([5,5,1,32]) # patch5*5 in size 1,out_size 32
    with tf.name_scope('bias1'):
        b_conv1 = bias_variable([32])
    # 第一层输出的内容 输出之前做非线性处理
    with tf.name_scope('h_conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) #输出的大小28*28*32
    # 池化
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  # 输出大小 14*14*32

"""
卷积第二层
"""
with tf.name_scope('layer2'):
    with tf.name_scope('weights2'):
        W_conv2 = weight_variable([5,5,32,64]) # patch5*5 in size 32,out_size 64
    with tf.name_scope('bias2'):
        b_conv2 = bias_variable([64])
    with tf.name_scope('h_conv1'):
       # 第一层输出的内容 输出之前做非线性处理
       h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) #输出的大小14*14*64
# 池化
    with tf.name_scope('h_pool1'):
        h_pool2 = max_pool_2x2(h_conv2)  # 输出大小 7*7*64
"""
func1 layer 定义神经网络layer 相当于全连接层
"""
#为第二层神经网络的size，变得更高1024
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

# 转换为一维数组
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
# 防止过拟合
# h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
"""
func2 layer 定义神经网络layer 相当于全连接层
"""
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
#softmax 分类 算概率
prediction = tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2)

"""
缩小误差
最小化误差用的损失函数，损失函数是目标类别和预测类别之间的交叉熵。
"""
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                             reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


"""
TensorFlow程序的流程是先创建一个图，在session中启动
"""
sess=tf.Session()
# writer = tf.train.SummaryWriter("logs/",sess.graph)
writer = tf.summary.FileWriter('./logs', sess.graph)     # write to file
merge_op = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())

"""
训练
"""
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print (compute_accuracy(mnist.test.images,mnist.test.labels))


