import tensorflow as tf
hello = tf.constant('It works')
sess = tf.Session()
print(sess.run(hello))
