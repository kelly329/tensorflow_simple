import tensorflow as tf

a = tf.Variable(tf.random_normal([5,10]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
