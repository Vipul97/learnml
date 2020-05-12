from ._base import LinearRegression
from ._stochastic_gradient import SGDClassifier, SGDRegressor
from ._ridge import Ridge
from ._perceptron import Perceptron

__all__ = ['LinearRegression',
           'Perceptron',
           'Ridge',
           'SGDClassifier',
           'SGDRegressor'
           ]
