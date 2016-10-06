import numpy as np
import tensorflow as tf
import math

class MMD():



    def __init__(self):

        print ('Calculating mmd...')



    def gaussiankernel(self,x,y,sigma):


        #get eculidean distance
        squared = tf.square(tf.sub(x, y))
        reduced = tf.reduce_sum(squared)
        distance = tf.to_float(reduced)
        #distance = tf.sqrt(converted)

        # need to estimate
        yita = tf.div(distance,tf.mul(tf.constant(-2.,dtype=tf.float32),sigma))

        # get gaussian kernel
        gaussian = tf.exp(yita)


        return gaussian



    def getkernel(self,input_x,input_y,n_source,n_target,dim,sigma):
        '''

        :param x: sourceMatrix
        :param y: targetMatrix
        :param n_source: # of source samples
        :param n_target: # of target samples
        :param dim: # of input dimension(features)
        :return: a scala showing the MMD
        '''
        # ---------------------------------------
        # x = tf.convert_to_tensor(input_x,dtype=tf.float32)
        # y = tf.convert_to_tensor(input_y, dtype=tf.float32)


        x = tf.cast(input_x,tf.float32)
        y = tf.cast(input_y, tf.float32)


        k_ss = k_st = k_tt = tf.constant(0.)
        n_ss = n_st = n_tt = tf.constant(0.)
        flag = tf.constant(1.)
        signal = tf.constant(-2.0)
        shape = [1,dim]
        for s in range(n_source):
            for s_ in range(n_source):
                list1 = tf.slice(x, [s, 0], shape)
                list2 = tf.slice(x, [s_, 0], shape)
                k_ss = tf.add(self.gaussiankernel(list1,list2,sigma),k_ss)
                n_ss = tf.add(n_ss,flag)


        for t in range(n_target):
            for t_ in range(n_target):
                list1 = tf.slice(y, [t, 0], shape)
                list2 = tf.slice(y, [t_, 0], shape)
                k_tt = tf.add(self.gaussiankernel(list1, list2, sigma), k_tt)
                n_st = tf.add(n_st, flag)


        for s in range(n_source):
            for t in range(n_target):
                list1 = tf.slice(x, [s, 0], shape)
                list2 = tf.slice(y, [t, 0], shape)
                k_st = tf.add(self.gaussiankernel(list1, list2, sigma), k_st)
                n_tt = tf.add(n_tt, flag)




        term1 = tf.div(k_ss,n_ss )
        term2 = tf.div( k_tt, n_tt)
        term3 = tf.mul(signal, tf.div(k_st,n_st))
        term4 = tf.add(term1,term2)

        kernel = tf.add(term3, term4)


        return kernel

    def getBandWidth(self,input_x,input_y,n_source,n_target,dim):
        '''
        gamma = 1/E(||x-y||)
        :param input_x:
        :param input_y:
        :param sigma:
        :param n_source:
        :param n_target:
        :param dim:
        :return: gamma
        '''
        x = tf.cast(input_x, tf.float32)
        y = tf.cast(input_y, tf.float32)
        counter = tf.constant(float(n_source))
        sum_up = tf.constant(.0)
        shape = [1, dim]
        for s in range(n_source):
            list1 = tf.slice(x, [s, 0], shape)
            list2 = tf.slice(y, [s, 0], shape)

            # get ||x-y||
            squared = tf.square(tf.sub(list1, list2))
            norm = tf.reduce_sum(tf.sqrt(squared))
            norm = tf.div(norm,tf.constant(float(dim)))

            sum_up  = tf.add(sum_up,tf.to_float(norm))


        gamma = tf.div(counter,sum_up)

        return gamma




    def getOptimizedKernel(self,input_x,input_y,n_source,n_target,dim,sigma):
        '''
                Note we could keep n_source and n_target with the same value and even!
                Optimized kernel, calculate in a linear time!
                :param x: sourceMatrix
                :param y: targetMatrix
                :param n_source: # of source samples
                :param n_target: # of target samples
                :param dim: # of input dimension(features)
                :return: a scala showing the MMD
                '''


        x = tf.cast(input_x, tf.float32)
        y = tf.cast(input_y, tf.float32)



        g = tf.constant(0.)



        n_samples = tf.constant(0.)
        flag = tf.constant(1.)
        signal = tf.constant(-1.0)

        shape = [1, dim]

        for i in range(1,math.floor(n_source/2)):
            # g_i = k(1,2)+k(3,4)-k(1,4)-k(2,3)
            #print('Now..%s'%i)
            list1 = tf.slice(x, [2 * i - 1, 0], shape)
            list2 = tf.slice(x, [2 * i, 0], shape)
            list3 = tf.slice(y, [2 * i - 1, 0], shape)
            list4 = tf.slice(y, [2 * i, 0], shape)

            z1 = tf.add(self.gaussiankernel(list1, list2, sigma), self.gaussiankernel(list3, list4, sigma))
            z2 = tf.add(self.gaussiankernel(list1, list4, sigma), self.gaussiankernel(list2, list3, sigma))
            z_i = tf.add(z1, tf.mul(signal, z2))

            n_samples = tf.add(n_samples, flag)

            g = tf.add(g,z_i)


        kernel = tf.div(g, n_samples)

        print('Getting MMD...')
        return kernel

    def getMMD(self,input_x,input_y,n_source,n_target,dim,num_kernel):
        '''
        Get MMD: a linear combination of kernel functions
        mmd = b1*k1 + b2*k2 + ... (dot product of beta and sigma)
        :param input_x: source
        :param input_y: target
        :param beta: [b1,b2,...] a list of beta (a tensor), need to optimize it during training
        :param sigma_square: [s1,s2,...] a list of beta (a tensor)
        :param n_source: number of samples in x
        :param n_target: number of samples in y
        :param dim: dimension
        :return: the value of mmd
        '''
        beta_size = (math.ceil(num_kernel/4) - math.floor(-num_kernel/4))*2
        beta = tf.constant(1 / beta_size, shape=[beta_size])

        sigma = self.getBandWidth(input_x, input_y, n_source, n_target, dim)

        # get many sigmas
        power = np.arange(math.floor(-num_kernel/4), math.ceil(num_kernel/4), 1/2)
        times = tf.constant(np.asarray([math.pow(2, i) for i in power]), dtype=tf.float32)

        mmd = tf.constant(0.)
        # slice 1-D tensor
        for i in range(beta_size):
            small_beta = tf.slice(beta,[i],[1])
            small_sigma = tf.mul(tf.slice(times,[i],[1]),sigma)
            kernel = self.getOptimizedKernel(input_x,input_y,n_source,n_target,dim,small_sigma)
            mmd = tf.add(mmd,tf.mul(kernel,small_beta))

        return mmd



    def testSession(self):
        # This is the test session
        num_samples = 90
        dimension = 64
        num_kernels = 10


        x = tf.random_normal([num_samples, dimension], mean=0.0,stddev=0.01)
        y = tf.random_normal([num_samples, dimension], mean=0.0,stddev=0.01)

        sigma = self.getBandWidth(x,y,num_samples,num_samples,dimension)

        kernel = self.getkernel(x, y, num_samples, num_samples, dimension, sigma)
        kernel_new = self.getMMD(x, y, num_samples, num_samples, dimension, num_kernels)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        s = sess.run(sigma)
        print('Sigma:',s)
        k = sess.run(kernel)
        print('Before Optimization MMD:', k)
        kn = sess.run(kernel_new)

        print('After Optimization MMD:', kn)


def main():

    mmd = MMD()

    print(mmd.testSession())


if __name__  == '__main__':
    main()

