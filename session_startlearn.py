import tensorflow as tf

# 1行2列
matrix1 = tf.constant([[3, 3]])
# 2行1列
matrix2 = tf.constant([[2],
                       [2]])
# 矩阵乘法  np.dot(m1,m2)
product = tf.matmul(matrix1, matrix2)

# 方法一
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

#方法二 运行到最后自动关闭
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

