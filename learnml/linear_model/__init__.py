from ._base import LinearRegression

from .stochastic_gradient import SGDRegressor
from .ridge import Ridge
from ._logistic import LogisticRegression
from ._perceptron import Perceptron

__all__ = ['LinearRegression',
           'LogisticRegression',
           'Perceptron',
           'Ridge',
           'SGDRegressor'
]
