import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 2, 2])
x = tf.Print(x, [x], message="P1")
i = tf.reshape(x, [-1, 4])
i = tf.Print(i, [i], message="P2")
i.eval(feed_dict={x: [[[1,2], [3,4]], [[5,6], [7,8]]]})