import tensorflow as tf 

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6 ,6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0,], [1, 0, 0], [1, 0, 0,]]

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, [None, 4])
    Y = tf.placeholder(tf.float32, [None, 3])
    nb_classes = 3

    W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

    hypothesis = tf.matmul(X, W) + b

    # tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    softmax = tf.nn.softmax(hypothesis)
    # Cross entropy cost/loss
    cost = tf.reduce_mean(- tf.reduce_sum(Y * tf.log(softmax), axis=1))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    # Launch graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            sess.run(train, feed_dict={
                X: x_data, Y: y_data
            })
            if step % 200 == 0:
                print(step, sess.run(cost, feed_dict={
                    X: x_data, Y: y_data
                }))

        all = sess.run(hypothesis, feed_dict={
            X: 
            [[1, 11, 7, 9],
            [1, 3, 4, 3],
            [1, 1, 0, 1]]
        })

        print(all, sess.run(tf.argmax(all, 1)))