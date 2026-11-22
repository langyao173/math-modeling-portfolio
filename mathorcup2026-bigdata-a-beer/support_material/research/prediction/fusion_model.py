"""Small pure-NumPy operations shared by inference and cache validation."""
import numpy as np

def combine_predictions(y16,native50,core_weight=.5,other_weight=.5):
    a,b=np.asarray(y16,float),np.asarray(native50,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=50:
        raise ValueError('Both branches must have shape (n,50).')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('A branch contains nonfinite predictions.')
    for w in [core_weight,other_weight]:
        if not 0<=w<=1:raise ValueError('Convex weights must be in [0,1].')
    weights=np.r_[np.full(30,core_weight),np.full(20,other_weight)]
    return weights*a+(1-weights)*b

def relative_grades(predictions,cuts):
    p,c=np.asarray(predictions,float),np.asarray(cuts,float)
    if c.shape!=(6,50) or not np.isfinite(c).all() or np.any(np.diff(c,axis=0)<0):
        raise ValueError('Expected six ordered finite cuts per target.')
    if p.ndim!=2 or p.shape[1]!=50 or not np.isfinite(p).all():
        raise ValueError('Expected finite (n,50) predictions.')
    return np.stack([np.searchsorted(c[:,j],p[:,j],side='right') for j in range(50)],axis=1)

def optimal_weight(y,a,b,indices):
    """Use only supplied training-side labels; caller controls the split."""
    y,a,b=np.asarray(y,float),np.asarray(a,float),np.asarray(b,float)
    if y.shape!=a.shape or a.shape!=b.shape:raise ValueError('Training shapes differ.')
    var=y.var(0)
    if np.any(var[indices]<=0):raise ValueError('A training target is constant.')
    da=a[:,indices]-b[:,indices]
    numerator=-np.mean((b[:,indices]-y[:,indices])*da/var[indices])
    denominator=np.mean(da*da/var[indices])
    return float(np.clip(numerator/denominator,0,1)) if denominator>1e-15 else .5
