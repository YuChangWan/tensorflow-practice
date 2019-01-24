import tensorflow as tf

if __name__ == '__main__':
    # X and Y data
    # if X==1 then Y == 2
    x_train = [1,2,3]
    y_train = [1,2,3]

    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32)
    

    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = X * W + b

    #cost/Loss function 
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    #Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    #Launch the graph in a session.
    sess = tf.Session()
    #Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    #Fit the Line.
    for step in range(2001):
        cost_v, W_v, b_v, _ = sess.run([cost,W,b,train], feed_dict={
            X: x_train, Y: y_train
        })
        if step % 20 == 0:
            print(step, cost_v, W_v, b_v)
    
    #Test        
    print( sess.run(hypothesis, feed_dict={X:[1,2,3,4,5,6]}))

