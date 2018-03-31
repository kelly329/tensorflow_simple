# -- coding: utf-8 --
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters 超参数 根据经验定的
lr = 0.001               # 学习率
training_iters = 100000  # 循环次数
batch_size = 128  # 每批处理的样本个数
'''
  mnist data input(image shape:28*28)  
  每一次输入为每一行的28个像素
'''
n_inputs= 28  # 每张图片每一行28个像素点
n_steps = 28  # time step 28行的像素
n_hidden_unis = 128 # neurons in hidden layer
n_classes = 10 # mnist classes (0-9digits)

'''
tf Graph input
'''
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

'''
define weights
'''
weights = {
    # (28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),
    # (128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
}
biases = {
    # (128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,])),
    # (10,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X, weights, baises):
    # hidden layer for input to cell
    ################################
    # X (128batch，28steps，28inputs) ==》（128*28,28inputs）
    X = tf.reshape(X,[-1,n_inputs])
    # ==> (128 batch*28 steps,128 hidden)
    X_in = tf.matmul(X,weights['in'])+biases['in']
    # ==> (128 batch,28 steps,128 hidden)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_unis])

    # cell
    ################################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True)
    # lstm cell 分成两个部分(c_state,m_state) c_state主线的state m_state 分线的state
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    # time_major time是否在第一主维度
    outputs,states= tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)

    # hidden layer for outputs as the final results
    ###############################################
    results = tf.matmul(states[1],weights['out']+baises['out'])

    return results




pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    with step * batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy,feed_dict={
                x:batch_xs,
                y:batch_ys
            }))
        step += 1

