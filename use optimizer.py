import tensorflow as tf 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #tf Graph Input
    X = [1,2,3]
    Y = [1,2,3]

    #Set wrong model weights
    W = tf.Variable(-5.0)
    #Linear model
    hypothesis = X * W
    #cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    #Minimize: Gradient Descent Magic
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    # Launch the graph in a session
    sess = tf.Session()
    #Initiallizes global variables in the graph
    sess.run(tf.global_variables_initializer())

    W_val = []
    cost_val = []
    for step in range(100):
        curr_W, curr_cost =  sess.run([W,cost])
        W_val.append(curr_W)
        cost_val.append(curr_cost)
        sess.run(train)

    plt.plot(W_val, cost_val, 'ro')
    plt.show()