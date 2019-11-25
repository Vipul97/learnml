from ._base import LinearRegression

from .stochastic_gradient import SGDClassifier, SGDRegressor
from .ridge import Ridge
from ._perceptron import Perceptron

__all__ = ['LinearRegression',
           'LogisticRegression',
           'Perceptron',
           'Ridge',
           'SGDClassifier',
           'SGDRegressor'
           ]
