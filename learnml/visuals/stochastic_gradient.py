from learnml.metrics import mean_squared_error
from learnml.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def plot_regression(model, X, y, poly_features=None, xlabel='x_1', ylabel='y'):
    plt.xlabel(f'${xlabel}$', fontsize=18)
    plt.ylabel(f'${ylabel}$', fontsize=18)

    X_new = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)

    if poly_features:
        X_new = poly_features.transform(X_new)

    y_new = model.predict(X_new)

    plt.plot(X, y, 'b.')
    plt.plot(X_new[:, 0], y_new, 'r-', linewidth=2, label='Predictions')
    plt.legend(loc='upper left', fontsize=14)
    plt.show()


def plot_w_path(w_path):
    plt.xlabel(r'$\theta_0$', fontsize=18)
    plt.ylabel(r'$\theta_1$', fontsize=18)
    plt.plot(w_path[:, 0], w_path[:, 1], 'r-s')
    plt.show()


def plot_learning_curves(model, X, y):
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.xlabel('Training set size', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.plot(np.sqrt(train_errors), 'r-', linewidth=2, label='train')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=2, label='val')
    plt.legend(loc='upper right', fontsize=14)
    plt.show()


def plot_logistic_regression(model, X, y):
    X_new = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
    y_proba = model.predict_proba(X_new)
    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

    plt.ylabel('Probability', fontsize=14)
    plt.plot(X[y == 0], y[y == 0], 'bs')
    plt.plot(X[y == 1], y[y == 1], 'g^')
    plt.plot([decision_boundary, decision_boundary], [0, 1], 'k:', linewidth=2)
    plt.plot(X_new, y_proba[:, 1], 'g-', linewidth=2)
    plt.plot(X_new, y_proba[:, 0], 'b--', linewidth=2)
    plt.text(decision_boundary + 0.02, 0.15, 'Decision  boundary', fontsize=14, color='k', ha='center')
    plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
    plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
    plt.axis([np.min(X), np.max(X), -0.02, 1.02])
    plt.show()


def plot_logistic_regression_contour(model, X, y):
    x0, x1 = np.meshgrid(
        np.linspace(np.min(X[:, 0]) - 0.02, np.max(X[:, 0]) + 0.02, 1000).reshape(-1, 1),
        np.linspace(np.min(X[:, 1]) - 0.02, np.max(X[:, 1]) + 0.02, 1000).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_proba = model.predict_proba(X_new)

    zz = y_proba[:, 1].reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

    left_right = np.array([np.min(X[:, 0]) - 0.02, np.max(X[:, 0]) + 0.02])
    boundary = -(model.coef_[0] * left_right + model.intercept_) / model.coef_[1]

    plt.clabel(contour, inline=1, fontsize=12)
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'bs')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'g^')
    plt.plot(left_right, boundary, 'k--', linewidth=3)
    plt.show()
