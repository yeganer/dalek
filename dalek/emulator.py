import numpy as np
import sklearn.decomposition
import sklearn.gaussian_process


class Emulator(object):

    def __init__(self, X, Y, n_components):
        # Use GaussianProcessRegressor later
        self._ncomp = n_components
        self._PCA = sklearn.decomposition.PCA(n_components=n_components)
        self._GP = sklearn.gaussian_process.GaussianProcess(
                theta0=1e-2,
                thetaL=1e-4,
                thetaU=1e-1,
                )
        self.train(X, Y)

    def train(self, X, Y):
        self._X = X
        self._Y = Y
        comp = self._PCA.fit_transform(Y)
        self._GP.fit(X, comp)

    def predict(self, X):
        X = np.atleast_2d(X)
        n_eval = X.shape[0]
        y = np.zeros((n_eval, self._GP.y.shape[1]))
        MSE = np.zeros(n_eval)
        for k in range(n_eval / 1000 + 1):
            batch_from = k * 1000
            batch_to = min([(k + 1) * 1000 + 1, n_eval + 1])
            y_, MSE_ = self._GP.predict(
                    X[batch_from:batch_to],
                    eval_MSE=True)
            #print(batch_from, batch_to, y_.shape)

            y[batch_from:batch_to] = y_
            MSE[batch_from:batch_to] = MSE_
        std = np.sqrt(MSE)
        return self.comp_to_spectrum(y), std

    def quick_predict(self, X):
        X = np.atleast_2d(X)
        assert X.shape[0] == 1
        return self.comp_to_spectrum(self._GP.predict(X))[0]

    def pca_transform(self, Y):
        return self._PCA.inverse_transform(self._PCA.transform(Y))

    def comp_to_spectrum(self, Y):
        return self._PCA.inverse_transform(Y)


class LogEmulator(Emulator):

    def train(self, X, Y):
        super(LogEmulator, self).train(X, np.log10(Y))

    def comp_to_spectrum(self, Y):
        return 10 ** super(LogEmulator, self).comp_to_spectrum(Y)


class DecoupledGaussianProcess(object):

    def __init__(self, N, *args, **kwargs):
        self.N = N
        self._GP = [
                sklearn.gaussian_process.GaussianProcess(*args, **kwargs)
                for _ in xrange(N)
                ]
        self.y = None

    def fit(self, X, y):
        assert y.shape[1] == self.N
        self.y = y
        for i in xrange(self.N):
            self._GP[i].fit(X, y[:, i])

    def predict(self, X, eval_MSE=False):
        X = np.atleast_2d(X)
        y = np.zeros((X.shape[0], self.N))
        if eval_MSE:
            MSE = np.zeros((X.shape[0], self.N))
            for i in xrange(self.N):
                y[:, i], MSE[:, i] = self._GP[i].predict(X, eval_MSE=eval_MSE)
            return y, MSE
        else:
            for i in xrange(self.N):
                y[:, i] = self._GP[i].predict(X)
            return y


    # def __getattr__(self, name):
    #     try:
    #         getattr(self._GP[0], name).__call__
    #     except AttributeError:
    #         return getattr(self._GP[0], name)
    #     else:
    #         def f(*args, **kwargs):
    #             for GP in self._GP:
    #                 getattr(GP, name)(*args, **kwargs)
    #         return f


class DecoupledEmulator(Emulator):

    def __init__(self, X, Y, n_components):
        # Use GaussianProcessRegressor later
        self._ncomp = n_components
        self._PCA = sklearn.decomposition.PCA(n_components=n_components)
        self._GP = DecoupledGaussianProcess(
                n_components,
                theta0=1e-2,
                thetaL=1e-4,
                thetaU=1e-1,
                storage_mode='light',
                )
        self.train(X, Y)

    def predict(self, *args, **kwargs):
        raise NotImplementedError
