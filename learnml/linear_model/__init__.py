from ._base import LinearRegression
from ._ridge import Ridge
from ._perceptron import Perceptron
from ._stochastic_gradient import SGDClassifier, SGDRegressor

__all__ = ['LinearRegression',
           'Perceptron',
           'Ridge',
           'SGDClassifier',
           'SGDRegressor'
           ]
