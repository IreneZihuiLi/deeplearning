
import tensorflow as tf
import numpy as np
import sys
from load_pic import load_test_pics,load_train_pics,load_batch


def load_writing(filename,headerfile):
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

    return header,id


# define hyper-parameters
num_hidden = 10
num_step = 1001
num_features = 192
num_class = 99
starter_learning_rate = 0.05

# cnn parameters
patch_size = 5
image_size = 300
num_channels = 1
depth = 16

# if type in number of step and hidden unit
if len(sys.argv) == 3:
    num_step = int(sys.argv[1]) + 1
    num_hidden = int(sys.argv[2])

batch_size = 594

with tf.device('/cpu:0'):
    # Placeholders, weights and biases
    x = tf.placeholder(tf.float32,
                       shape=(batch_size,image_size,image_size, num_channels))
    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_class], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_class]))

    # Model.

    def model(data):
        # computing
        conv = tf.nn.conv2d(x, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        logits = tf.matmul(hidden, layer4_weights) + layer4_biases
        return logits

    logits = model(x)

    # this is for write outputs into files
    probs = tf.nn.softmax(logits)

    # the true tag goes here
    y_ = tf.placeholder(tf.float32, [batch_size, num_class])
    # define cross entropy as cost function
    # cross_entropy = -tf.reduce_sum(y_*tf.log(hidden+0.001))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_) )

    # Learning rate decay
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                            1000, 0.96, staircase=True)
    # minimize cross_entropy using the Adam algorithm with a learning rate of 0.0.
    train_step = tf.train.GradientDescentOptimizer(starter_learning_rate).minimize(cross_entropy)

    # launch the model in a Session, run the initialized operation
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    # evaluation
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    # define inputs
    test_x = load_test_pics()
    data_x, data_y = load_train_pics()

    # define write out file
    header, id = load_writing('data/test.csv', 'data/sample_submission.csv')

    # train_step feeding in the batches data to replace the placeholders
    for i in range(num_step):
        batch_x,batch_y = load_batch(data_x,data_y,batch_size)
        _,loss,sh,acc = sess.run([train_step,cross_entropy,logits,accuracy], feed_dict={x: batch_x, y_: batch_y})

        print(i, loss, acc)



        if i % 500 ==0 :
            # write results
            submission = open('data/subcnn.csv', 'w')
            submission.write(header)
            results = sess.run(probs, feed_dict= {x:test_x})
            for index,item in enumerate(results):
                submission.write(id[index]+',')
                strings = [str(x) for x in item]
                submission.write(','.join(strings)+'\n')

            print ('Finished!')

            submission.close()