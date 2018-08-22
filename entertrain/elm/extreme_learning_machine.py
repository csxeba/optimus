from keras.models import Model
from keras.layers import Input
from entertrain import backend as E


class ExtremeLearningMachine:

    def __init__(self,
                 feature_extractor: Model,
                 predictor: Model,
                 solver="pseudoinverse"):
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.full_model = None  # type: Model
        self._predictor_bias = self.predictor.get_weights()[-1]
        self.solver = {"pseudoinverse": self._solve_pseudoinverse,
                       "covariance": self._solve_covariance}[solver]
        self._build_full_model()

    def _build_full_model(self):
        model_input = Input(batch_shape=self.feature_extractor.input_shape)
        model_output = self.predictor(self.feature_extractor(model_input))
        self.full_model = Model(inputs=model_input, outputs=model_output)

    @classmethod
    def from_single_model(cls, model: Model, solver="pseudoinverse"):
        fe_input = Input(batch_shape=model.input_shape)
        fe_out = model.layers[-2](fe_input)
        feature_extractor = Model(inputs=fe_input, outputs=fe_out)
        pr_in = Input(batch_shape=model.layers[-2].output_shape)
        pr_out = model(pr_in)
        predictor = Model(inputs=pr_in, outputs=pr_out)
        return cls(feature_extractor, predictor, solver)

    def _solve_pseudoinverse(self, Z, Y):
        W = E.dot(E.pinv(Z), Y)
        self.predictor.set_weights([W, self._predictor_bias])

    def _solve_covariance(self, Z, Y):
        A = E.cov(Z.T)
        B = E.cov(Z.T, Y.T)
        W = E.dot(E.inv(A), B)
        self.predictor.set_weights([W, self._predictor_bias])

    def train_on_batch(self, x, y):
        z = self.feature_extractor.predict(x)
        self.solver(z, y)

    def predict(self, x):
        z = self.feature_extractor.predict(x)
        return self.predictor.predict(z)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1,
                 sample_weight=None, steps=None, loss=None, **kwargs):
        if not self.full_model._is_compiled:
            if loss is None:
                raise RuntimeError("Model needs to be built for evaluation, please supply a loss!")
            self.full_model.compile(optimizer="sgd", loss=loss, **kwargs)
        return self.full_model.evaluate(x, y, batch_size, verbose, sample_weight, steps)
