
# coding: utf-8

# In[15]:

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from numpy import genfromtxt




# # convert ndarray to tensor
# def my_func(arg):
#   arg = tf.convert_to_tensor(arg, dtype=tf.float32)
# #   sess=tf.InteractiveSession()
# #   print arg.eval()
#   return arg

# Load in training and testing data
# Shape of (record_num, col_num): (999,7)
my_data = genfromtxt('output.csv', delimiter=',')
my_testing = genfromtxt('testing.csv', delimiter=',')

category = 2




# In[8]:



# batch function
def my_batch(category,num,my_data):
#     choose a num of batch return x, y as vectors
    size,col = my_data.shape
    idList = random.sample(range(0, size), num )
    batch_xs = np.zeros(shape=(num,col-category))
    batch_ys = np.zeros(shape=(num,category))
    
    for index,i in enumerate(idList):
        line = my_data[i,:-category]
        batch_xs[index]=line
        label = my_data[i,col-category:col]
        batch_ys[index]=label
    return batch_xs,batch_ys
    
def testingDataLoader(category,my_data):
    size,col = my_data.shape
    
    batch_xs = np.zeros(shape=(size,col-category))
    batch_ys = np.zeros(shape=(size,category))
    for index,line in enumerate(my_data):
        batch_xs[index]=line[:-category]
        batch_ys[index]=line[col-category:col]

        
    return batch_xs,batch_ys


# In[ ]:




# In[9]:

# Implementation starts!

# None means any number, so x is not a specific number here.
x = tf.placeholder(tf.float32, [None, 6],name="x-input")


# In[16]:

# Init weights, bias, (all zeros first) and define softmax function
W = tf.Variable(tf.zeros([6, 2]),name="weights")
b = tf.Variable(tf.zeros([2],name="bias"))

# use a name scope to organize nodes in the graph visualizer
with tf.name_scope("Wx_b") as scope:
# first multiply x and w, then add b vector. apply softmax to get probabilities
  y = tf.nn.softmax(tf.matmul(x, W) + b)
#     y = tf.matmul(x, W) + b
print "y",type(y)

# Add summary ops to collect data
w_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)


weights_summary = tf.histogram_summary("weights", W)
biases_summary = tf.histogram_summary("biases", b)
y_summary = tf.histogram_summary("y", y)


# In[20]:

# Trainning
y_ = tf.placeholder(tf.float32, [None, 2],name="y-input")

with tf.name_scope("xent") as scope:
# tf.log computes logarithm of each element. 
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
with tf.name_scope("train") as scope:
# minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.01. 
  train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)


# In[21]:



# In[10]:



# In[22]:

with tf.name_scope("test") as scope:
# evaluation
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  accuracy_summary = tf.scalar_summary("accuracy", accuracy)

  
  
init = tf.initialize_all_variables()

# launch the model in a Session, run the initialized operation
sess = tf.Session()
sess.run(init)


  
# Merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.merge_summary([weights_summary, biases_summary, y_summary])
writer = tf.train.SummaryWriter("/tmp/read2", sess.graph_def)
tf.train.SummaryWriter("/tmp/read2", sess.graph_def).flush()
# tf.train.SummaryWriter.flush()



# In[11]:



# get testing data
test_xs,test_ys = testingDataLoader(category,my_testing)


# Train for 1000 times!
# batch of 100 at each time
# train_step feeding in the batches data to replace the placeholders
for i in range(1000):
  if i % 10 == 0:
    feed = {x: test_xs,y_:test_ys}
    result = sess.run([merged, accuracy], feed_dict=feed)
    summary_str = result[0]
    acc = result[1]
    writer.add_summary(summary_str, i)
    print("Accuracy at step %s: %s" % (i, acc))
  else:
    batch_xs, batch_ys = my_batch(category,100, my_data)
    feed = {x: batch_xs, y_: batch_ys}
    sess.run(train_step, feed_dict=feed)




# In[12]:


  
    
# test_xs,test_ys = testingDataLoader(category,my_testing)
print sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys})


# In[13]:




# In[ ]:



