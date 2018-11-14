from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.utils import shuffle

import itertools

import pandas as pd
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["area", "beds", "price"]
FEATURES = ["area", "beds"]
LABEL = "price"

training_set = pd.read_csv("house.csv", skipinitialspace=True,
                skiprows=1, names=COLUMNS)

# Model parameters
W = tf.Variable([[0.1], [0.1]], dtype=tf.float32)
b = tf.Variable([1], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = tf.matmul(x, W) + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.00000001)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
x_train = pd.DataFrame({k: training_set[k].values for k in FEATURES})
y_train = pd.Series(training_set[LABEL].values)
print(y_train)
print(x_train)
idx = np.random.permutation(x_train.index)

for j in range(5000):
    for i in range(0, 5):
	X = x_train.iloc[i:i+1, 0:2]
	Y = y_train.iloc[i]	
        sess.run(train, {x: X, y: Y})
	#print(sess.run(x_train.iloc[[i]]))
    idx = np.random.permutation(x_train.index)
    x_train.reindex(idx).reset_index(drop = True)
    y_train.reindex(idx).reset_index(drop = True)

# evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: X, y: Y})
    if (j % 10 == 0): 
	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

x_predict = pd.DataFrame([[2000, 2]])
print(sess.run(linear_model, {x: x_predict}))
