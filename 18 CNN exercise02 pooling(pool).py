import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

image = np.array([[[[4], [3]],
                   [[2], [1]]]], dtype=np.float32)

if __name__ == '__main__':
    pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                          strides=[1, 1, 1, 1], padding='SAME')
    sess = tf.InteractiveSession()
    pool_img = pool.eval()
    print(pool_img.shape)
    print(pool_img)

    plt.imshow(pool_img.reshape(2,2), cmap='Greys')
    plt.show()
