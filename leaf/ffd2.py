
import tensorflow as tf
import numpy as np
import sys


# data helper functions
def load_training(filename):

    f = open(filename, 'r')
    x = []
    y = []

    for line in f.readlines():
        seq = line.split(',')

        features = [float(x) for x in seq[:-1]]
        x.append(features)
        y.append(int(seq[-1].strip()))

    n_values = np.max(y) + 1
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


# define hyper-parameters
num_hidden = 10
num_step = 5001
num_features = 192
num_class = 99
starter_learning_rate = 0.5

if len(sys.argv) == 3:
    num_step = int(sys.argv[1]) + 1
    num_hidden = int(sys.argv[2])


# Placeholders, weights and biases
x = tf.placeholder(tf.float32, [None, num_features])
W_1 = tf.Variable(tf.zeros([num_features, num_hidden]))
b_1 = tf.Variable(tf.zeros([num_hidden]))
W_2 = tf.Variable(tf.zeros([num_hidden, num_class]))
b_2 = tf.Variable(tf.zeros([num_class]))

hidden = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)
hidden = tf.matmul(hidden,W_2) + b_2
probs = tf.nn.softmax(hidden)

# the true tag goes here
y_ = tf.placeholder(tf.float32, [None, num_class])
# define cross entropy as cost function
# cross_entropy = -tf.reduce_sum(y_*tf.log(hidden+0.001))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hidden,y_) )

# Learning rate decay
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.96, staircase=True)
# minimize cross_entropy using the Adam algorithm with a learning rate of 0.0.
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# launch the model in a Session, run the initialized operation
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# evaluation
correct_prediction = tf.equal(tf.argmax(hidden,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# define inputs
test_xs, test_ys = load_training('data/tftest.csv')
batch_xs, batch_ys = load_training('data/ctrain.csv')
header,id,dev = load_test('data/test.csv','data/sample_submission.csv')



# train_step feeding in the batches data to replace the placeholders
for i in range(num_step):

    _,loss = sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        acc,h = sess.run([accuracy,hidden], feed_dict={x: test_xs, y_: test_ys})
        print (i, loss, acc)

    if i %500 ==0:
        submission = open('data/sub.csv', 'w')
        submission.write(header)
        results = sess.run(probs, feed_dict= {x:dev})
        for index,item in enumerate(results):
            submission.write(id[index]+',')
            strings = [str(x) for x in item]
            submission.write(','.join(strings)+'\n')


        print ('Finished!')

        submission.close()