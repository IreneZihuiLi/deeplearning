import tensorflow as tf
import numpy as np
# import seaborn
import matplotlib.pyplot as plt

# For constants, we do not need to initialize them
print("----------------Constants----------------")
a = tf.constant(3.5)
b = tf.constant(4.)

c = a * b

with tf.Session() as sess:
    print (sess.run(c))
    print (c.eval())


# For Variables, they are in-memory buffers, we need to initialize them
print("----------------Variables----------------")
w1 = tf.ones((2,2))
w2 = tf.Variable(tf.zeros((2,2)),name="Weights")

with tf.Session() as sess:
    print (sess.run(w1))
    sess.run(tf.initialize_all_variables())
    print (sess.run(w2))

# Updating
print("----------------Updating----------------")
state = tf.Variable(0,name="Counter")
new_value = tf.add(state,tf.constant(1))
update = tf.assign(state,new_value)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print (sess.run(state))
    for _ in range(4):
        sess.run(update)
        print (sess.run(state))


# Updating
print("----------------Fetching----------------")
a = tf.constant(3.5)
b = tf.constant(4.)

c = tf.add(a,b)
mul = tf.mul(a,c)

with tf.Session() as sess:
    res = sess.run([c,mul])
    print (res)

# Import from Numpy
print("-----------Import from Numpy-------------")
a = np.ones((2,2))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print (sess.run(ta))

print("-----------Placeholders Feed dicts-------------")
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
output = tf.mul(a,b)
with tf.Session() as sess:
    print (sess.run([output],feed_dict={a:[7.],b:[2.4]}))

# Variable scope
print("-----------Variable Scope-------------")
a = np.ones((2,2))
ta = tf.convert_to_tensor(a)
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v",[1])
assert v.name == "foo/bar/v:0"

# Reuse paras: like in RNN, sharing W
print("-----------Sharing Params-------------")
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    tf.get_variable_scope().reuse_variables() # Get the v in the same variable scope.
    v1 = tf.get_variable("v", [1])
assert v1 == v

