# Team review and independent work must be documented truthfully.
"""Fold-local sparse chemical regressors and training-only text supervision.

This is an independent extension of the MathorCup analysis. It does not execute
the authors' notebook or use held-out sensory values as input features.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

Q1 = Path(__file__).resolve().parents[2] / 'q1'
if str(Q1) not in sys.path:
    sys.path.insert(0, str(Q1))
from sensory_model import ChemicalPreprocessor
MODEL_VERSION = 'exact_penalized_rrr_v2'


def specifications():
    specs = []
    for k in [8, 24, 64, 231]:
        for alpha in [3., 30., 300.]:
            specs.append(dict(family='sparse_ridge', k=k, alpha=alpha))
    for k in [8, 24, 64, 231]:
        for gamma in [.25, 1.]:
            for alpha in [.1, 1.]:
                specs.append(dict(family='sparse_rbf', k=k, gamma=gamma, alpha=alpha))
    for k in [24, 64, 231]:
        for C in [.3, 1., 3.]:
            specs.append(dict(family='sparse_svr', k=k, C=C))
    for weight in [0., .5, 1.]:
        for rank in [5, 15, 30]:
            for alpha in [100., 300., 1000.]:
                specs.append(dict(family='auxiliary_rrr', weight=weight, rank=rank, alpha=alpha))
    return specs


class AdaptiveChemicalRegressor:
    def __init__(self, spec):
        self.spec = dict(spec)

    def fit(self, X, Y, W=None):
        self.prep_ = ChemicalPreprocessor('rank', 'median').fit(X)
        Z = self.prep_.transform(X)
        self.y_mean_ = Y.mean(axis=0)
        self.y_scale_ = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
        T = (Y - self.y_mean_) / self.y_scale_
        self.n_outputs_ = T.shape[1]
        s = self.spec
        if s['family'] == 'auxiliary_rrr':
            aug = T
            if s['weight'] > 0:
                if W is None:
                    raise ValueError('Auxiliary text is required for fitting, never for prediction.')
                wi = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(W)
                wi = np.sqrt(np.maximum(wi, 0.))
                wi = StandardScaler().fit_transform(wi)
                aug = np.column_stack([T, s['weight'] * np.sqrt(T.shape[1] / wi.shape[1]) * wi])
            gram = Z @ Z.T
            dual = cho_solve(cho_factor(gram + s['alpha'] * np.eye(len(Z))), aug)
            # Exact rank-constrained ridge: top eigenspace of
            # Y_aug.T X (X.T X + alpha I)^-1 X.T Y_aug.
            cross = aug.T @ gram @ dual
            _, vectors = np.linalg.eigh((cross + cross.T) / 2.)
            v = vectors[:, -min(s['rank'], len(vectors)):]
            self.coef_ = Z.T @ dual @ v @ v[:self.n_outputs_].T
            return self
        corr = Z.T @ T / max(len(Z) - 1, 1)
        self.keep_ = np.argsort(-np.abs(corr), axis=0, kind='stable')[:min(s['k'], Z.shape[1])].T
        self.fits_ = []
        for j, keep in enumerate(self.keep_):
            zz = Z[:, keep]
            if s['family'] == 'sparse_ridge':
                beta = cho_solve(cho_factor(zz.T @ zz + s['alpha'] * np.eye(len(keep))), zz.T @ T[:, j])
                self.fits_.append(beta)
            elif s['family'] == 'sparse_rbf':
                K = rbf_kernel(zz, gamma=s['gamma'] / len(keep))
                dual = cho_solve(cho_factor(K + s['alpha'] * np.eye(len(K))), T[:, j])
                self.fits_.append((zz, dual))
            elif s['family'] == 'sparse_svr':
                self.fits_.append(SVR(C=s['C'], epsilon=.1, kernel='rbf', gamma=1. / len(keep)).fit(zz, T[:, j]))
            else:
                raise ValueError(s['family'])
        return self

    def predict(self, X):
        Z = self.prep_.transform(X)
        s = self.spec
        if s['family'] == 'auxiliary_rrr':
            pred = Z @ self.coef_
        else:
            pred = np.empty((len(Z), self.n_outputs_))
            for j, (keep, fitted) in enumerate(zip(self.keep_, self.fits_)):
                zz = Z[:, keep]
                if s['family'] == 'sparse_ridge':
                    pred[:, j] = zz @ fitted
                elif s['family'] == 'sparse_rbf':
                    train, dual = fitted
                    pred[:, j] = rbf_kernel(zz, train, gamma=s['gamma'] / len(keep)) @ dual
                else:
                    pred[:, j] = fitted.predict(zz)
        return pred * self.y_scale_ + self.y_mean_


class PerTargetSelection:
    """One fixed candidate per output, selected using training OOF values."""
    def __init__(self, specs, indices):
        self.specs = specs
        self.indices = np.asarray(indices, int)

    def fit(self, X, Y, W=None):
        self.models_ = {int(i): AdaptiveChemicalRegressor(self.specs[i]).fit(X, Y, W)
                        for i in np.unique(self.indices)}
        if hasattr(self, 'output_indices_'):
            del self.output_indices_
        return self

    def compact_for_inference(self):
        """Keep only the fitted target states actually used by this selector.

        No parameter is refitted.  Original output indices are retained so the
        compact candidate predictions can be restored to the complete output
        order.  Calling this method twice is harmless; old uncompressed
        instances continue to use the original prediction path below.
        """
        if hasattr(self, 'output_indices_'):
            return self
        outputs = {int(i): np.flatnonzero(self.indices == i)
                   for i in np.unique(self.indices)}
        if set(outputs) != set(self.models_):
            raise ValueError('Candidate models do not match the selected indices.')
        for i in outputs:
            model = self.models_[i]
            if model.n_outputs_ != len(self.indices):
                raise ValueError('Expected an uncompressed model with every output.')
        for i, selected in outputs.items():
            model = self.models_[i]
            model.y_mean_ = model.y_mean_[selected].copy()
            model.y_scale_ = model.y_scale_[selected].copy()
            if model.spec['family'] == 'auxiliary_rrr':
                model.coef_ = model.coef_[:, selected].copy()
            else:
                model.keep_ = model.keep_[selected].copy()
                model.fits_ = [model.fits_[int(j)] for j in selected]
            model.n_outputs_ = len(selected)
        self.output_indices_ = outputs
        return self

    def predict(self, X):
        values = {i: model.predict(X) for i, model in self.models_.items()}
        if hasattr(self, 'output_indices_'):
            result = np.empty((len(X), len(self.indices)))
            for i, columns in self.output_indices_.items():
                result[:, columns] = values[i]
            return result
        return np.column_stack([values[int(i)][:, j] for j, i in enumerate(self.indices)])
