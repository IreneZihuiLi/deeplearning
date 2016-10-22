from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import numpy as np


def use_LogR(X_train, X_test, y_train, y_test):


    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    LogisticRegression(penalty='L2',max_iter=2)
    score = lr.score(X_test,y_test)
    return (score)

def use_SVM(X_train, X_test, y_train, y_test):
    clf = SVC()

    clf.fit(X_train,y_train)
    SVC(C=1.8, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    score = clf.score(X_test,y_test)
    return (score)



# load files
def load_data(filename):
    f = open(filename,'r')
    x = []
    y = []

    for line in f.readlines():
        seq = line.split(',')

        features = [float(x) for x in seq[:-1]]
        x.append(features)
        y.append(int(seq[-1].strip()))

    return x, y



x_train, y_train = load_data('data/tftrain.csv')

x_test, y_test = load_data('data/tftest.csv')



n_values = len(set(y_train))
print (np.shape(y_train),n_values)

print (np.max(y_train),np.min(y_train))

for index,item in enumerate(set(y_train)):
    print (item)

    if index % 10 == 0:
        print('\n')

# test for y
m = np.eye(n_values)[y_train]
print (np.shape(m))

#
# print (use_LogR(x_train, x_test, y_train, y_test))
# print (use_SVM(x_train, x_test, y_train, y_test))