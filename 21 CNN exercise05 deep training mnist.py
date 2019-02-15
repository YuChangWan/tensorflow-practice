import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
keep_prob = tf.placeholder(tf.float32)

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, [None, 784])
    X_img = tf.reshape(X, [-1, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])

    # L1 Img_In shape=(?, 28, 28, 1)
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    #   Conv    -> (?, 28, 28, 32)
    #   Pool    -> (?, 14, 14, 32)
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    print(L1)
    L1 = tf.nn.relu(L1)
    print(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    print(L1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    print(L1)

    # L2 Img_In shape=(?, 14, 14, 32)
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    #   Conv    -> (?, 14, 14, 64)
    #   Pool    -> (?, 7, 7, 64)
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    print(L2)
    L2 = tf.nn.relu(L2)
    print(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    print(L2)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    print(L2)

    # L3 Img_In shape=(?, 7, 7, 64)
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    #   Conv    -> (?, 7, 7, 128)
    #   Pool    -> (?, 4, 4, 128)
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    print(L3)
    L3 = tf.nn.relu(L3)
    print(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    print(L3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    print(L3)
    
    # spread it to be fully-connected
    L3 = tf.reshape(L3, [-1, 4*4*128])
    print(L3)

    # L4 FC 4*4*128 inputs -> 625 outputs
    W4 = tf.get_variable(
        'W4', shape=[4 * 4 * 128, 625], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([625]))
    L4 = tf.matmul(L3, W4) + b4
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
    print(L4)
    
    # L5 finial FC 625 inputs -> 10 outputs
    W5 = tf.get_variable(
        'W5', shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer()
    )
    b5 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(L4,W5) + b5
    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)

    # initialize
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # train model
        print('Learning started. It takes sometime.')
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
                c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch

            print('Epoch: {:04d} cost = {:.9f}'.format(epoch + 1, avg_cost))

        print('Learning Finished!')

        # Test model and check accuracy
        correct_prediction = tf.equal(
            tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuraacy:', sess.run(accuracy, feed_dict={
            X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1
        }))
