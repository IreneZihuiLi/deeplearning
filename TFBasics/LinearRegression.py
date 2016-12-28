# Code from https://www.youtube.com/watch?v=L8Y2_Cq2X5s&index=8&list=PLCJlDcMjVoEdtem5GaohTC1o9HTTFtK7_

# A linear regression example
print("-----------Linear Regression-------------")
# create data points
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)

# set sizes
n_samples = 1000
batch_size = 100

# reshape into (n,1) for TensorFlow
X_data = np.reshape(X_data,(n_samples,1))
y_data = np.reshape(y_data,(n_samples,1))

# placeholders for inputs
X = tf.placeholder(tf.float32,shape=(None,1))
y = tf.placeholder(tf.float32,shape=(None,1))

with tf.variable_scope("linear-regression"):
    # reuse = False, so these tensors are created anew
    W = tf.get_variable("weights",(1,1),initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1, ), initializer=tf.constant_initializer(0))
    # +b:it is doing casting itself!
    y_pred = tf.matmul(X,W) + b 
    # Average Square Error
    loss = tf.reduce_sum((y-y_pred)**2/n_samples)

    # run one step of GD
    #TF scope is not python scope: so loss here is visible
    opt_operation = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
    #init
    sess.run(tf.initialize_all_variables())
    #GD loop for 5000 steps
    for i in range(5000):
        #Select random minibatch
        indices = np.random.choice(n_samples,batch_size)
        X_batch, y_batch = X_data[indices],y_data[indices]
        _, loss_val = sess.run([opt_operation,loss],feed_dict={X:X_batch,y:y_batch})
        if (i %100) == 0 :
            print (loss_val,i)
    # get predicted
    y_predicted = sess.run(y_pred,feed_dict={X:X_data,y:y_data})

# plot results
plt.scatter(X_data,y_predicted,color='red')
plt.scatter(X_data, y_data,color='blue')
plt.show()


