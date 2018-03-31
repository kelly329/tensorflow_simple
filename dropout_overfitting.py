# -- coding: utf-8 --
import tensorflow as tf

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

'''load data 加载数据'''
digits = load_digits()
X = digits.data
y = digits.target
# 将标签二值化 如：yes 1 no 0
y = LabelBinarizer().fit_transform(y)
# 划分训练集与测试集  验证集占训练集的30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''定义层'''
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weigths = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weigths) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + 'outputs', outputs)
    return outputs

'''定义占位符作为神经网络的输入'''
keep_prob = tf.placeholder(tf.float32) # 保持多少不被dropout
xs = tf.placeholder(tf.float32,[None,64]) # 8*8
ys = tf.placeholder(tf.float32,[None,10])

'''添加输出层'''
l1 = add_layer(xs, 64, 50,'l1',activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10 ,'l2', activation_function=tf.nn.softmax)

'''the loss between prediction and real data'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
'''记录summary'''
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
test_writer = tf.summary.FileWriter("logs/test",sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})
    if i % 50 == 0:
        # record loss 在记录时保持不dropout任何东西
        train_result = sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result = sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)




