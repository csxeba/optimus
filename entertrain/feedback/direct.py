from keras import Model
from keras import backend as K
from keras.optimizers import Optimizer

from ..backend import uniform


class DirectFeedbackAlignment(Optimizer):

    def __init__(self, params, output_shape, **kwargs):
        super().__init__(**kwargs)
        feedback_weights = [
            K.variable(uniform(-1, 1, size=output_shape[1:]))
            for layer in params
        ]
        loss_derivative = K.gradients(self.model.loss, self.model.output)
        updates = [K.dot(loss_derivative, W) for W in feedback_weights]
        self.update_getters = [K.function()]

    def get_updates(self, loss, params):
        pass


class DirectFeedbackAligment:

    def __init__(self, model):
        self.model = model  # type: Model
        feedback_weights = [K.variable(
            uniform(-1, 1, size=layer.output_shape[1:]))
            for layer in self._trainable_layers()
        ]
        loss_derivative = K.gradients(self.model.loss, self.model.output)
        updates = [K.dot(loss_derivative, W) for W in feedback_weights]
        self.update_getters = [K.function()]

    def _trainable_layers(self):
        return [layer for layer in self.model.layers if layer.trainable]

    def train_on_batch(self, x, y):
        pred = self.model.predict(x)
        delta = self.model.loss(y, pred)
