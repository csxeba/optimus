from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils.np_utils import normalize, to_categorical

from entertrain.elm import ExtremeLearningMachine


def pull_mnist():
    (lX, lY), (tX, tY) = mnist.load_data()
    lX, tX = map(normalize, (lX, tX))
    lY, tY = map(lambda y: to_categorical(y, num_classes=10), (lY, tY))
    return lX.reshape(-1, 784), lY, tX.reshape(-1, 784), tY


def build_net(inshape, outshape):
    ann = Sequential([
        Dense(128, activation="tanh", input_shape=inshape),
        Dense(outshape[0], activation="linear")
    ])
    return ann


def xperiment():
    lX, lY, tX, tY = pull_mnist()
    ann = build_net(lX.shape[1:], tX.shape[1:])
    elm = ExtremeLearningMachine.from_single_model(ann)
    elm.train_on_batch(lX, lY)
    elm.evaluate(tX, tY, loss="mse", metrics=["acc"])


if __name__ == '__main__':
    xperiment()
