"""Chemical-to-sensory models with fold-local preprocessing.

 Teams must independently review model choices and all results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import joblib


class ChemicalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, mode='rank', imputer='median'):
        self.mode = mode
        self.imputer = imputer

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.first_imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
        A = self.first_imputer_.fit_transform(X)
        if self.mode == 'rank':
            self.sorted_ = np.sort(A, axis=0)
            Z = self._rank_scores(A)
            self.scaler_ = StandardScaler().fit(Z)
        else:
            self.scaler_ = StandardScaler().fit(A)
        if self.imputer == 'knn':
            Z = self.scaler_.transform(self._rank_scores(A) if self.mode == 'rank' else A)
            Z[np.isnan(X)] = np.nan
            self.knn_ = KNNImputer(n_neighbors=7, weights='distance', keep_empty_features=True).fit(Z)
        return self

    def _rank_scores(self, A):
        n = len(self.sorted_)
        Z = np.empty_like(A)
        for j in range(A.shape[1]):
            lo=np.searchsorted(self.sorted_[:,j],A[:,j],side='left')
            hi=np.searchsorted(self.sorted_[:,j],A[:,j],side='right')
            Z[:,j]=norm.ppf((lo+hi+1.)/(2.*(n+1.)))
        return Z

    def transform(self, X):
        X = np.asarray(X, float)
        A = self.first_imputer_.transform(X)
        Z = self.scaler_.transform(self._rank_scores(A) if self.mode == 'rank' else A)
        if self.imputer == 'knn':
            Z[np.isnan(X)] = np.nan
            Z = self.knn_.transform(Z)
        return Z


class SensoryEstimator:
    def __init__(self, spec, imputer='median'):
        self.spec = dict(spec)
        self.imputer = imputer

    def fit(self, X, Y):
        s = self.spec
        self.prep_ = ChemicalPreprocessor(s.get('mode', 'rank'), self.imputer).fit(X)
        Z = self.prep_.transform(X)
        self.y_mean_ = np.mean(Y, axis=0)
        self.y_std_ = np.std(Y, axis=0, ddof=1)
        self.y_std_[self.y_std_ < 1e-10] = 1
        T = (Y-self.y_mean_)/self.y_std_
        if s['family'] in ('ridge_raw', 'ridge_rank'):
            self.est_ = Ridge(alpha=s['alpha'])
        elif s['family'] == 'rbf':
            self.est_ = KernelRidge(alpha=s['alpha'], kernel='rbf', gamma=s['gamma']/Z.shape[1])
        elif s['family'] == 'pls':
            self.est_ = PLSRegression(n_components=s['n_components'], scale=False, max_iter=500)
        elif s['family'] == 'extra_trees':
            self.est_ = ExtraTreesRegressor(n_estimators=240, max_features=s['max_features'],
                min_samples_leaf=s['min_samples_leaf'], n_jobs=1, random_state=20261023)
        elif s['family'] == 'random_forest':
            self.est_ = RandomForestRegressor(n_estimators=240, max_features=s['max_features'],
                min_samples_leaf=s['min_samples_leaf'], n_jobs=1, random_state=20261023)
        else:
            raise ValueError(s['family'])
        self.est_.fit(Z, T)
        return self

    def predict(self, X):
        return np.asarray(self.est_.predict(self.prep_.transform(X))).reshape(len(X), -1)*self.y_std_+self.y_mean_


def specifications():
    specs=[]
    for fam,mode in [('ridge_raw','raw'),('ridge_rank','rank')]:
        for a in [10.,30.,100.,300.,1000.]:
            specs.append(dict(family=fam,mode=mode,alpha=a))
    for a,g in [(.1,.1),(1.,.1),(1.,.5),(10.,.5)]:
        specs.append(dict(family='rbf',mode='rank',alpha=a,gamma=g))
    for nc in [2,5,10]:
        specs.append(dict(family='pls',mode='rank',n_components=nc))
    for mf,leaf in [(.35,2),(.7,2),(.7,5)]:
        specs.append(dict(family='extra_trees',mode='raw',max_features=mf,min_samples_leaf=leaf))
    for mf,leaf in [(.5,2),(.7,5)]:
        specs.append(dict(family='random_forest',mode='raw',max_features=mf,min_samples_leaf=leaf))
    return specs


def learn_weights(preds, Y):
    scale=np.std(Y,axis=0,ddof=1)
    P=np.stack(preds,axis=-1)
    k=P.shape[-1]
    def loss(w):
        return float(np.mean(((np.einsum('ijk,k->ij',P,w)-Y)/scale)**2)+.02*np.sum((w-1/k)**2))
    r=minimize(loss,np.ones(k)/k,method='SLSQP',bounds=[(0.,1.)]*k,
               constraints={'type':'eq','fun':lambda w:w.sum()-1.},options={'ftol':1e-10,'maxiter':300})
    if not r.success:
        raise RuntimeError(r.message)
    return r.x


def grades_from_reference(pred, reference):
    """Relative grades (0..6); never an inverse of per-taster normalization."""
    cuts=np.quantile(reference,np.arange(1,7)/7,axis=0)
    return np.stack([np.searchsorted(cuts[:,j],pred[:,j],side='right') for j in range(pred.shape[1])],axis=1)


def predict_bundle(bundle, frame, include_grades=False):
    if isinstance(bundle,(str,Path)):
        bundle=joblib.load(bundle)
    X=frame[bundle['features']].to_numpy(float) if isinstance(frame,pd.DataFrame) else np.asarray(frame,float)
    P=sum(w*m.predict(X) for w,m in zip(bundle['weights'],bundle['models']))
    ans=pd.DataFrame(P,columns=bundle['targets'])
    if include_grades:
        return ans,pd.DataFrame(grades_from_reference(P,bundle['reference_y']),columns=bundle['targets'])
    return ans
