from keras import backend as K
from theano import tensor as T


def pseudoinverse(X):
    return T.nlinalg.pinv(X)
