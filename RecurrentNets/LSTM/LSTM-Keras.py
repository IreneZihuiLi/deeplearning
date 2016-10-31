
# coding: utf-8


from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import numpy as np
import random
import sys



import load_data
print ('Now loading data for training')
X, y,maxlen,chars = load_data.get_batched()
print ('X',np.shape(X))
print ('y',np.shape(y))


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars)))) #(seq_len, voc_len)
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    # sample an index form a multinomial distribution
    # (n,pvals,size): number of examples, prob dist, output shape
    # argmax: get
    return np.argmax(np.random.multinomial(1, a, 1))

# >>> res = np.random.multinomial(1, [0.01,0.03,0.06,0.9], size=1)
# >>> res
# array([[0, 0, 0, 1]])
# >>> np.argmax(res)
# 3


# train the model, output generated text after each iteration
for iteration in range(1, 10):
    print()
    print('-' * 50)
    #print('Iteration', iteration)

    # training: X_train, Y_train
    model.fit(X, y, batch_size=1, nb_epoch=1)

    '''
    start_index = random.randint(0, len(text) - maxlen - 1)
    print ('**** start:',start_index)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]

        generated += sentence
        # a series for start

        print('----- Generating with seed: "' + sentence + '"')
        # sentence is a part of real sample : abor and that will oppose persistence, t
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))

            for t, char in enumerate(sentence):
                print ('input',char) #input o
                x[0, t, char_indices[char]] = 1.
            print ('Shape of testing x',np.shape(x))
            preds = model.predict(x, verbose=0)[0]
            # preds: prediction [0.01822178,0.01933114,...]  Voc_size
            #print ('preds',np.shape(preds),preds)
            next_index = sample(preds, diversity)
            #print ('Sampled in',next_index)
            #print ('next_index',np.shape(next_chars),next_chars)
            #get the sampled char
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
            #print ('**** sentence **** at No.',i,sentence)
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    '''
   


X_test, y_test, true_ids = load_data.load_test(chars,1)

for index,x in enumerate(X_test):
   #print ('shape of x',np.shape(x))
   seq = x
   prediction_result = []
   print ('Inputs,', true_ids[index])
   #print('At test sample %s, Shape of testing x %s' % (index, np.shape(x)))
   for i in range(5):


       new_x = np.zeros((1, maxlen, len(chars)))
       for t, id in enumerate(seq):
           #print('input', id)  # input o
           new_x[0, t, id] = 1.
       #new_x = np.expand_dims(x, axis=0)
       #print ('x is ',np.shape(new_x))
       preds = model.predict(new_x, verbose=0)
       #print ('Predictions ',np.shape(preds)) # shape [1,137]: [1, voc_size]
       # preds [0] -> [137,]
       predict_id = np.argmax(preds)
       #print ('Inputs', true_ids[index])
       #print ('Predictions',predict_id)
       prediction_result.append(predict_id)

       # create new series
       m = seq[1:]
       m.append(predict_id)
       seq = m
       #print('shape of seq', np.shape(seq))

   print ('Predictions,',prediction_result)





