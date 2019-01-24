import tensorflow as tf 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_data = [1,2,3]
    y_data = [1,2,3]

    W = tf.Variable(tf.random_normal([1]), name='weight')
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    #Our hypothesis for linear model X * W
    hypothesis = X * W

    #cost/loss function
    cost = tf.reduce_sum(tf.square(hypothesis - Y))

    #Minimize: Gradient Descent using derivative: W -= learning rate * derivate
    learning_rate = 0.1
    gradient = tf.reduce_mean((W * X - Y) * X)
    descent = W - learning_rate * gradient
    update = W.assign(descent)


    W_val = []
    cost_val = []
    #Launch the graph in a session.
    sess = tf.Session()
    #Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    for step in range(21):
        curr_cost, curr_W, _ = sess.run([cost, W, update],  feed_dict={X: x_data, Y: y_data})
        #print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        
        W_val.append(curr_W)
        cost_val.append(curr_cost)

    #Plot W - cost graph, 
    plt.plot(W_val, cost_val, 'ro')
    plt.axis([0, 2, 0, 1])
    plt.show()
