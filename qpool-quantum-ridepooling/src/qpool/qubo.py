"""Exact packing QUBO compilation, Ising conversion and raw-result validation.

Convention: E(x)=x.T@Q@x with upper triangular Q; H(s)=-s.T@J@s.
An additional reference spin is always used, x_i=(s_i*s_ref+1)/2.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
import csv, json
import numpy as np

@dataclass
class PackingInstance:
    weights: np.ndarray
    edges: list[tuple[int,int]]
    labels: list[dict] | None = None

    def __post_init__(self):
        self.weights=np.asarray(self.weights,dtype=float)
        if self.weights.ndim != 1 or not np.all(np.isfinite(self.weights)):
            raise ValueError('Weights must be a finite vector.')
        if np.any(self.weights < 0):
            raise ValueError('This compiler requires nonnegative optional-action gains.')
        n=len(self.weights)
        es=set()
        for u,v in self.edges:
            if not isinstance(u,(int,np.integer)) or not isinstance(v,(int,np.integer)):
                raise ValueError('Conflict indices must be integers.')
            if u==v or min(u,v)<0 or max(u,v)>=n:
                raise ValueError('Invalid conflict edge.')
            es.add((min(u,v),max(u,v)))
        self.edges=sorted(es)
        if self.labels is not None and len(self.labels)!=n:
            raise ValueError('One label per variable is required.')

    def adjacency(self):
        a=[set() for _ in self.weights]
        for i,j in self.edges:a[i].add(j);a[j].add(i)
        return a

    def evaluate(self,x):
        x=np.asarray(x)
        if x.shape!=self.weights.shape or not np.isin(x,[0,1]).all():
            raise ValueError('Invalid binary result dimension/domain.')
        conflicts=[(i,j) for i,j in self.edges if x[i] and x[j]]
        return {'reward':float(self.weights@x),'feasible':not conflicts,
                'n_conflicts':len(conflicts),'conflicts':conflicts}

    def to_dict(self):
        return {'weights':self.weights.tolist(),'edges':self.edges,'labels':self.labels}

    @classmethod
    def from_dict(cls,d):return cls(np.array(d['weights']),d['edges'],d.get('labels'))

def packing_qubo(instance:PackingInstance,margin:float=1.0):
    """P_ij=min(w_i,w_j)+margin gives strict one-flip feasibility."""
    if margin<=0:raise ValueError('A strictly positive penalty margin is required.')
    Q=np.diag(-instance.weights)
    for i,j in instance.edges:Q[i,j]=min(instance.weights[i],instance.weights[j])+margin
    return Q

def to_upper(Q):
    Q=np.asarray(Q,dtype=float)
    if Q.ndim!=2 or Q.shape[0]!=Q.shape[1] or not np.isfinite(Q).all():
        raise ValueError('Q must be finite and square.')
    return np.diag(np.diag(Q))+np.triu(Q+Q.T,1)

def qubo_to_ising(Q):
    Q=to_upper(Q); n=len(Q)
    cross=Q+Q.T-2*np.diag(np.diag(Q))
    J=np.zeros((n+1,n+1))
    J[:n,:n]=-cross/8.0
    h=np.diag(Q)/2.0+cross.sum(axis=1)/4.0
    J[:n,n]=-h/2;J[n,:n]=-h/2
    bias=float(np.trace(Q)/2.0+np.triu(Q,1).sum()/4.0)
    return J,bias

def ising_to_qubo(J):
    J=np.asarray(J,float)
    if J.ndim!=2 or J.shape[0]!=J.shape[1] or not np.allclose(J,J.T):
        raise ValueError('J must be square and symmetric.')
    n=len(J)-1
    A=J[:n,:n].copy();np.fill_diagonal(A,0)
    Q=np.zeros((n,n));np.fill_diagonal(Q,4*A.sum(axis=1)-4*J[:n,n])
    Q+=np.triu(-8*A,1)
    bias=float(-np.sum(A)+2*np.sum(J[:n,n])-np.trace(J))
    # H(s)=x^T Q x + bias. Diagonal J contributes only a constant.
    return Q,bias

def decode_spins(spins,n):
    s=np.asarray(spins)
    if s.shape!=(n+1,) or not np.isin(s,[-1,1]).all():
        raise ValueError(f'Expected {n+1} spins, all -1/+1, including reference spin.')
    return ((s[:-1]*s[-1]+1)//2).astype(int)

def exact_integer_ising(J,max_denominator=100000):
    """No coefficient rounding: find an exact rational common scale or reject."""
    J=np.asarray(J,float)
    fs=[]
    for v in J.flat:
        f=Fraction(float(v)).limit_denominator(max_denominator)
        if abs(float(f)-v)>1e-11:raise ValueError('Coefficient lacks supported exact rational representation.')
        fs.append(f)
    den=reduce(lcm,(f.denominator for f in fs),1)
    if den>10**12:raise ValueError('Common denominator too large; reformulate instead of silent rounding.')
    ints=[f.numerator*(den//f.denominator) for f in fs]
    divisor=reduce(gcd,(abs(v) for v in ints),0) or 1
    ints=[v//divisor for v in ints]
    if max(map(abs,ints),default=0)>2**53:raise ValueError('Integer coefficients too large.')
    return np.array(ints,dtype=np.int64).reshape(J.shape),den/divisor

def preflight_ising(J,limit=550):
    J=np.asarray(J)
    issues=[]
    if J.ndim!=2 or J.shape[0]!=J.shape[1]:return {'valid':False,'issues':['not_square']}
    if not np.isfinite(J).all():issues.append('nonfinite')
    if len(J)>limit:issues.append('spin_capacity_exceeded')
    if not np.allclose(J,J.T,atol=1e-12,rtol=0):issues.append('not_symmetric')
    if np.any(np.diag(J)!=0):issues.append('nonzero_diagonal')
    if not np.allclose(J,np.rint(J),atol=1e-12,rtol=0):issues.append('noninteger')
    if J.min(initial=0)<-128 or J.max(initial=0)>127:issues.append('signed_8bit_exceeded')
    return {'valid':not issues,'issues':issues,'n_spins':len(J),
            'minimum':float(J.min(initial=0)),'maximum':float(J.max(initial=0)),
            'nonzero_upper':int(np.count_nonzero(np.triu(J,1)))}

def compile_instance(instance,margin=1.0):
    Q=packing_qubo(instance,margin);J,bias=qubo_to_ising(Q)
    I,scale=exact_integer_ising(J)
    pf=preflight_ising(I)
    return Q,I,{'ising_scale':scale,'qubo_bias':bias,'penalty_margin':margin,
                'energy_relation':'E_qubo = H_ising / ising_scale + qubo_bias',
                'ising_energy_convention':'H=-s.T@J@s','qubo_convention':'E=x.T@Q@x',
                'reference_spin':'last','preflight':pf}

def write_matrix(path,matrix):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(path,np.asarray(matrix),delimiter=',',fmt='%.12g')

def read_matrix(path):
    a=np.loadtxt(path,delimiter=',',ndmin=2)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all():
        raise ValueError('Numeric CSV must be square, finite, with no header or row index.')
    return a

def export_instance(instance,folder,name,extra=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    Q,I,meta=compile_instance(instance)
    if not meta['preflight']['valid']:
        raise ValueError('Not hardware-ready: '+','.join(meta['preflight']['issues']))
    write_matrix(folder/(name+'_qubo.csv'),Q)
    write_matrix(folder/(name+'_ising.csv'),I)
    meta.update(instance.to_dict());meta.update(extra or {})
    meta['name']=name;meta['n_binary']=len(Q)
    meta['portal_csv_schema_verified']=False
    (folder/(name+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return meta

def repair(instance,x):
    """A separately reported deterministic repair; never changes the raw result."""
    x=np.asarray(x);before=instance.evaluate(x);x=x.astype(int).copy()
    adj=instance.adjacency()
    while True:
        pairs=[(i,j) for i,j in instance.edges if x[i] and x[j]]
        if not pairs:break
        nodes=set(k for edge in pairs for k in edge)
        u=min(nodes,key=lambda i:(instance.weights[i]/max(1,sum(x[j] for j in adj[i])),i))
        x[u]=0
    after=instance.evaluate(x)
    return x,{'raw':before,'repaired':after,'reward_removed':before['reward']-after['reward']}

def exact_enumeration(instance):
    n=len(instance.weights)
    if n>24:raise ValueError('Enumeration limited to 24 variables.')
    states=((np.arange(2**n,dtype=np.uint32)[:,None]>>np.arange(n))&1).astype(np.int8)
    good=np.ones(len(states),bool)
    for i,j in instance.edges:good&=~((states[:,i]==1)&(states[:,j]==1))
    rewards=states@instance.weights
    opt=float(np.max(rewards[good]));winners=states[good&(np.abs(rewards-opt)<1e-9)]
    return opt,winners

def connected_components(instance):
    adj=instance.adjacency();remaining=set(range(len(adj)));comps=[]
    while remaining:
        root=min(remaining);stack=[root];remaining.remove(root);comp=[]
        while stack:
            v=stack.pop();comp.append(v)
            for u in sorted(adj[v]&remaining):remaining.remove(u);stack.append(u)
        comps.append(sorted(comp))
    return comps

def induced(instance,indices):
    ids=list(indices);lookup={v:i for i,v in enumerate(ids)}
    es=[(lookup[u],lookup[v]) for u,v in instance.edges if u in lookup and v in lookup]
    labels=None if instance.labels is None else [instance.labels[i] for i in ids]
    return PackingInstance(instance.weights[ids],es,labels)

def safe_lns(instance,incumbent,free_vertices):
    """Condition on fixed selected actions; delete incompatible free alternatives.
    Returns an exact representation of that conditional neighborhood, not the full problem.
    """
    incumbent=np.asarray(incumbent)
    if not instance.evaluate(incumbent)['feasible']:raise ValueError('Incumbent must be feasible.')
    free=set(free_vertices);adj=instance.adjacency()
    fixed={i for i in np.flatnonzero(incumbent) if i not in free}
    blocked=set().union(*(adj[i] for i in fixed)) if fixed else set()
    kept=sorted(free-blocked)
    return induced(instance,kept),kept,sorted(fixed)
