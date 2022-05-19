import pandas as pd
import numpy as np
import sklearn.gaussian_process
import sklearn.model_selection
import sklearn.preprocessing

class Gaussian_Process_Regressor:
    def __init__(self, kernel_length_scale=1, n_restarts_optimizer=0):
        self.input_norm = sklearn.preprocessing.StandardScaler()
        self.output_norm = sklearn.preprocessing.StandardScaler()
        self.model = sklearn.gaussian_process.GaussianProcessRegressor(
            # kernel=sklearn.gaussian_process.kernels.RBF(
            #     length_scale=kernel_length_scale
            # ),
            kernel=sklearn.gaussian_process.kernels.Matern(nu=kernel_length_scale),
            n_restarts_optimizer=n_restarts_optimizer
        )

    def train(self, x, y, y_err=None):
        x = np.array(x)
        y = np.array(y)
        # self.model.set_params(self, alpha=y_err**2)
        self.model.set_params(alpha=0.15)
        x = self.input_norm.fit_transform(x)
        y = self.output_norm.fit_transform(y)
        self.model.fit(x, y)

    def predict(self, x):
        x = np.array(x)
        pred = self.model.predict(self.input_norm.transform(x))
        pred = self.output_norm.inverse_transform(pred)
        return pred

    def score(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x = self.input_norm.transform(x)
        y = self.output_norm.transform(y)
        return self.model.score(x, y)

    def cross_validation_scores(self, x, y, cv=5):
        x = np.array(x)
        y = np.array(y)
        x = self.input_norm.fit_transform(x)
        y = self.output_norm.fit_transform(y)
        print(x.mean(axis=0), y.mean(), x.std(axis=0), y.std())
        scores = sklearn.model_selection.cross_val_score(
            self.model, x, y, cv=cv
        )
        print(scores)
        return scores.mean()

    def loss(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return np.mean((self.predict(x) - y) ** 2)

def kernel_length_scale_search(search_set, x, y, cv=5):
    score = -np.inf
    params = None
    scores = []
    # print("X shape", x.shape)
    # print("Y shape", y.shape)
    for s in search_set:
        gpr = Gaussian_Process_Regressor(
            kernel_length_scale=s,
            # n_restarts_optimizer=3
        )
        new_score = gpr.cross_validation_scores(x, y, cv=cv)

        # gpr.train(x[:idx], y[:idx])
        # new_score = gpr.score(x[idx:], y[idx:])
        # print(new_score)
        scores.append(new_score)
        if new_score > score:
            score = new_score
            params = s
    return params, scores
