import numpy as np 

if __name__ == '__main__':
    xy = np.loadtxt('./data/data-01-test-score.csv',delimiter=',', dtype=np.float32)
    x_data = xy[:, 0:-1]
    y_data = xy[:,-1:]

    print(x_data.shape, x_data, len(x_data))
    print(y_data.shape, y_data)