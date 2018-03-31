import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 获取数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义神经网络层
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 定义占位符
# 不规定输入的图片个数，784像素点
xs = tf.placeholder(tf.float32,[None,784]) # 28*28
ys = tf.placeholder(tf.float32,[None,10]) # 10个输出

# 添加输出层
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    # 学习一部分data
    batch_xs,batch_ys = mnist.train.next_batch(100)
    if i% 50 == 0:
        sess.run()

