# -- coding: utf-8 --
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.arg_max(y_pre,1),tf.arg_max(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

'''
定义卷积层
x即为图片的所有信息，w为weight strides步长
strides[1,x_movement,y_movement,1]
padding 是否卷积改变大小  vaild改变
'''
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

'''
定义池化层 max_pooling
ksize 池化窗口的大小 四维向量[1,height,width,1] 卷积核的大小
ksize不想在batch和channels上操作，所以这两个维度上设为1
第一个参数x的shape为[batch,height,width,channels]
第三个参数和卷积类似，窗口在每一个维度上滑动的步长，一般为[1,stride,stride,1]
'''
def max_pool_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 占位符
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
# -1是不用管这个维度的大小，reshape会自动计算，把数据扁平化，1 channel 灰白
# -1地方计算出来就是samples的说少，导入例子的多少
x_image= tf.reshape(xs,[-1,28,28,1])
# print(x_image.shape)  #[n_samples,28,28,1]


'''
定义层conv1 layer
'''
W_conv1 = weight_variable([5,5,1,32]) #patch 5*5 in size 1,out_size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) # output size 28*28*32
h_pool1 = max_pool_2X2(h_conv1) # output size 14*14*32

'''
定义层conv2 layer
'''
W_conv2 = weight_variable([5,5,32,64]) #patch 5*5 in size 32,out_size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) # output size 14*14*64
h_pool2 = max_pool_2X2(h_conv2) # output size 7*7*64

'''
func1 layer 即全连接层
'''
# in_size 7*7*64，out_size 1024 更厚
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) # [n_samples,7,7,64]-->[n_samples,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
# 解决过拟合
# h_fc1_drop= tf.nn.dropout(h_fc1,keep_prob)

'''
func2 layer 即全连接层
'''
# in_size 7*7*64，out_size 1024 更厚
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

#softmax 分类 算概率
prediction = tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    # 每次训练多少个图像
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))