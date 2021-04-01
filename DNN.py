import tensorflow as tf
import numpy as np
import pandas as pd
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

xy=np.loadtxt("C:/Users/박준형/Desktop/fds.csv",delimiter=',', dtype=np.float32)
df = pd.read_csv("C:/Users/박준형/Desktop/fds.csv",encoding='utf-8')
xy = MinMaxScaler(xy)
train_set, test_set = split_train_test(xy, 0.3)
print(len(train_set), "train +", len(test_set), "test")
x_train = train_set[:,0:-1]
y_train = train_set[:,[-1]]
x_test = test_set[:,0:-1]
y_test = test_set[:,[-1]]
learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, 30])
Y = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([30, 25]))
b1 = tf.Variable(tf.random_normal([25]))
L1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([25, 20]))
b2 = tf.Variable(tf.random_normal([20]))
L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([20, 15]))
b3 = tf.Variable(tf.random_normal([15]))
L3 = tf.nn.sigmoid(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(tf.random_normal([15, 10]))
b4 = tf.Variable(tf.random_normal([10]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)


W5 = tf.Variable(tf.random_normal([10, 1]))
b5 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.nn.sigmoid(tf.matmul(L4, W5) + b5)

cost = tf.losses.sigmoid_cross_entropy(Y, hypothesis)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    _, c = sess.run([optimizer, cost],feed_dict={X:x_train, Y:y_train, keep_prob:0.5})
    if step %100==0:
        print("Step : ", step, " Cost : ", c)   
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 0.5}))