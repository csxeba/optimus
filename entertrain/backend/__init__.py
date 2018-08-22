from ..config import config

if config.backend == "numpy":
    print("EnterTrain: Using NumPy backend")
    from .numpy_backend import *
else:
    assert False
