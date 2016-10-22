import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
import tensorflow as tf
import numpy as np
import sys

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def loading(filename):

    f = open(filename, 'r')
    x = []
    y = []

    for line in f.readlines():
        seq = line.split(',')

        features = [float(x) for x in seq[:-1]]
        x.append(features)
        y.append(int(seq[-1].strip()))

    n_values = np.max(y) + 1
    # test for y
    y_expand = np.eye(n_values)[y]

    return np.asarray(x), y_expand

def load_test(filename,headerfile):
    f = open(filename, 'r')
    x = []
    id = []

    for line in f.readlines():
        seq = line.split(',')
        if seq[0]!='id':
            features = [float(x) for x in seq[1:]]
            x.append(features)
            id.append(seq[0])

    headerf = open(headerfile, 'r')
    header = ''
    for line in headerf.readlines():
        seq = line.split(',')
        if seq[0]== 'id':
            header = line

    return header,id,np.asarray(x)

# None means any number, so x is not a specific number here.
x = tf.placeholder(tf.float32, [None, 192])


# Init weights, bioas, (all zeros first) and define softmax function
W = tf.Variable(tf.zeros([192, 99]))
b = tf.Variable(tf.zeros([99]))

# first multiply x and w, then add b vector. apply softmax to get probabilities
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Trainning
y_ = tf.placeholder(tf.float32, [None, 99])


cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.01.
train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

init = tf.initialize_all_variables()
# launch the model in a Session, run the initialized operation
sess = tf.Session()
sess.run(init)


# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

test_xs, test_ys = loading('data/tftest.csv')
batch_xs, batch_ys = loading('data/ctrain.csv')

header,id,dev = load_test('data/test.csv','data/sample_submission.csv')


num_step = 3001
if len(sys.argv) > 1:
    num_step = int(sys.argv[1]) + 1

# batch of 100 at each time
# train_step feeding in the batches data to replace the placeholders
for i in range(num_step):

    _,loss, outputs = sess.run([train_step,cross_entropy,y], feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        acc,results = sess.run([accuracy,y], feed_dict={x: test_xs, y_: test_ys})
        print (i, loss, acc)

    if i % 500 == 0 :
        submission = open('data/sub.csv', 'w')
        submission.write(header)
        results = sess.run(y, feed_dict= {x:dev})
        print (np.shape(results))
        for index,item in enumerate(results):
            submission.write(id[index]+',')
            strings = [str(x) for x in item]
            submission.write(','.join(strings)+'\n')


        print ('Finished!')

        submission.close()

