import tensorflow as tf 
import numpy as np 

file_path = './data/data-04-zoo.csv'

if __name__ == '__main__':
    xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
    x_data = xy[:, :-1]
    y_data = xy[:, -1:]

    nb_classes = 7
    nb_Xs = 16

    X = tf.placeholder(tf.float32, [None, nb_Xs])
    Y = tf.placeholder(tf.int32, (None, 1))
    # one_hot expands one rank more -> [None, 1, 7] ex) [[[1,0,0,0,0,0,0]], [[0,1,0,0,0,0,0]], ...]
    Y_one_hot = tf.one_hot(Y, nb_classes) 
    # So we need to reshape it.
    # -1 means don't touch -> [None, 7] ex) [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], ...]
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) 

    W = tf.Variable(tf.random_normal([nb_Xs, nb_classes]), name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

    hypothesis = tf.matmul(X,W) + b

    # cross-entropy cost/loss

    # - tf.reduce_sum(Y * tf.log(hypothesis), axis=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot)
    cost = tf.reduce_mean(cross_entropy)
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction,tf.arg_max(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            sess.run(train, feed_dict={X: x_data, Y: y_data})
            if step % 100 == 0 :
                loss, yy, acc = sess.run([cost, Y,accuracy], feed_dict={
                    X: x_data, Y: y_data
                    })
                print("Step: {:5}\tLoss: {:.3f}\tAcc:{:.2%}".format(
                    step, loss, acc))
        #Let's see if we can predict
        pred = sess.run(prediction, feed_dict={X: x_data})

        #y_data: (N, 1) = flatten => (N, ) matches pred.shape
        for p, y in zip(pred, y_data.flatten()):
            print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
