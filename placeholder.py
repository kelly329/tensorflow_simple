import tensorflow as tf

# input1 = tf.placeholder(tf.float32,[2,2])

# 有placeholder占位符 需要配套使用feed_dict 字典
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))
