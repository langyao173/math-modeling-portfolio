import itertools,json,ast
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import linprog
from qpool.qubo import *
from qpool.solvers import *

@pytest.mark.parametrize('seed',range(16))
def test_qubo_conversion_exact(seed):
    rng=np.random.default_rng(seed);n=1+seed%8
    raw=rng.integers(-7,8,(n,n));Q=to_upper(raw)
    J,bias=qubo_to_ising(Q);Q2,b2=ising_to_qubo(J)
    assert np.allclose(Q,Q2) and abs(bias+b2)<1e-10
    for bits in itertools.product((0,1),repeat=n):
        x=np.array(bits);s=np.r_[2*x-1,1]
        assert np.isclose(x@raw@x,x@Q@x)
        assert np.isclose(x@Q@x,-s@J@s+bias)
        assert np.array_equal(decode_spins(-s,n),x)
    J+=np.eye(n+1)*3
    Q3,b3=ising_to_qubo(J)
    assert np.allclose(Q3,Q) and np.isclose(b3,b2-3*(n+1))

@pytest.mark.parametrize('seed',range(20))
def test_penalty_local_minima_and_milp(seed):
    rng=np.random.default_rng(seed);n=8;w=rng.integers(0,10,n)
    edges=[(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<.4]
    inst=PackingInstance(w,edges);Q=packing_qubo(inst,.25)
    states=np.array(list(itertools.product((0,1),repeat=n)));E=np.einsum('bi,ij,bj->b',states,Q,states)
    brute,winners=exact_enumeration(inst)
    assert np.isclose(E.min(),-brute)
    for x,e in zip(states,E):
        if not inst.evaluate(x)['feasible']:
            assert any((lambda y:y@Q@y)(x^np.eye(n,dtype=int)[i])<e-1e-8 for i in range(n))
    x,info=solve_milp(inst,3)
    assert info['optimal'] and np.isclose(w@x,brute)

@pytest.mark.parametrize('seed',range(12))
def test_tv_oracle_and_coupled_master(seed):
    rng=np.random.default_rng(seed);m=5;p=rng.dirichlet(np.ones(m));z=rng.normal(size=m);radius=seed/12
    q,val=tv_worst_probability(p,z,radius)
    C=np.r_[z,np.zeros(m)]
    A=np.vstack([np.c_[np.eye(m),-np.eye(m)],np.c_[-np.eye(m),-np.eye(m)],np.r_[np.zeros(m),np.ones(m)]])
    b=np.r_[p,-p,2*radius]
    lp=linprog(C,A_ub=A,b_ub=b,A_eq=np.array([np.r_[np.ones(m),np.zeros(m)]]),b_eq=[1],bounds=(0,None),method='highs')
    assert lp.success and abs(val-lp.fun)<1e-8
    assert abs(q.sum()-1)<1e-10 and np.min(q)>=-1e-10 and np.sum(abs(q-p))/2<=radius+1e-8
    n=6;inst=PackingInstance(rng.random(n),[(0,1),(1,2),(3,4),(2,5)])
    Z=rng.normal(size=(m,n));x,info=solve_joint_tv(inst,Z,p,radius,.7,time_limit=5,max_iter=80)
    opt=max(inst.weights@np.array(bits)+.7*tv_worst_probability(p,Z@np.array(bits),radius)[1]
            for bits in itertools.product((0,1),repeat=n) if inst.evaluate(bits)['feasible'])
    assert abs(info['robust_value']-opt)<1e-6
    assert info['robust_upper_bound']>=opt-1e-7


def test_joint_vs_separable_counterexample():
    # Joint robust hedging preserves diversification; sum of singleton worst cases destroys it.
    inst=PackingInstance([0,0,0],[(0,2),(1,2)])
    Z=np.array([[1,0,.4],[0,1,.4]]);p=np.array([.5,.5])
    x,info=solve_joint_tv(inst,Z,p,.5,1,3)
    sep=np.array([tv_worst_probability(p,Z[:,i],.5)[1] for i in range(3)])
    y,_=solve_milp(PackingInstance(sep,inst.edges))
    assert np.array_equal(x,[1,1,0]) and np.array_equal(y,[0,0,1])
    assert np.isclose(info['robust_value'],1)
    assert np.isclose(tv_worst_probability(p,Z@y,.5)[1],.4)


def test_preflight_boundaries_and_rejection():
    inst=PackingInstance(np.ones(549),[]);Q,J,meta=compile_instance(inst)
    assert J.shape==(550,550) and meta['preflight']['valid']
    assert not compile_instance(PackingInstance(np.ones(550),[]))[2]['preflight']['valid']
    dense=PackingInstance(np.ones(200),list(itertools.combinations(range(200),2)))
    assert not compile_instance(dense)[2]['preflight']['valid']
    with pytest.raises(ValueError):packing_qubo(inst,0)
    with pytest.raises(ValueError):decode_spins(np.zeros(4),3)
    with pytest.raises(ValueError):PackingInstance([-1],[])
    assert not preflight_ising(np.array([[0,200],[200,0]]))['valid']
    assert not preflight_ising(np.array([[0,.1],[.1,0]]))['valid']


def test_safe_neighborhood_and_extra_conflicts():
    inst=PackingInstance([5,4,3,2,1],[(0,1),(1,2),(2,3),(3,4)])
    inc=np.array([1,0,1,0,1]);sub,keep,fixed=safe_lns(inst,inc,[1,2,3])
    assert keep==[2] and fixed==[0,4]
    x,_=solve_milp(sub);z=np.zeros(5,int);z[fixed]=1;z[keep]=x
    assert inst.evaluate(z)['feasible']
    labels=[{'vehicle':i,'requests':[i]} for i in range(3)]
    other=PackingInstance([4,3,2],[(0,1)],labels)
    y,_=solve_milp(other);assert other.evaluate(y)['feasible'] and other.weights@y==6
    # Non-clique groups must not create constraints absent from the declared graph.
    other=PackingInstance([1,1,1],[(0,1)],[{'vehicle':0,'requests':[]}]*3)
    y,_=solve_milp(other);assert other.weights@y==2


def test_raw_repair_and_ising_export(tmp_path):
    inst=PackingInstance([5,4,3],[(0,1),(1,2)])
    x,report=repair(inst,[1,1,1]);assert not report['raw']['feasible'] and report['repaired']['feasible']
    meta=export_instance(inst,tmp_path,'QP_TEST03')
    assert not meta['portal_csv_schema_verified']
    J=read_matrix(tmp_path/'QP_TEST03_ising.csv');Q=read_matrix(tmp_path/'QP_TEST03_qubo.csv')
    for bits in itertools.product((0,1),repeat=3):
        s=np.r_[2*np.array(bits)-1,1]
        assert np.isclose(np.array(bits)@Q@np.array(bits),-s@J@s/meta['ising_scale']+meta['qubo_bias'])


def test_python310_grammar():
    root=Path(__file__).resolve().parents[1]
    for path in list((root/'src').rglob('*.py'))+list((root/'scripts').glob('*.py')):
        ast.parse(path.read_text(),filename=str(path),feature_version=(3,10))

def test_sdk_documented_conversion_example():
    Q=np.array([[1.,1.],[0.,1.]])
    J,bias=qubo_to_ising(Q)
    assert np.allclose(J,[[0,-.125,-.375],[-.125,0,-.375],[-.375,-.375,0]])
    assert bias==1.25


def test_result_parser_distinguishes_raw_repaired(tmp_path):
    from qpool.results import inspect_solutions,read_solutions
    inst=PackingInstance([5,4,3],[(0,1),(1,2)])
    export_instance(inst,tmp_path,'QP_TEST03')
    rows=np.array([[1,1,1,1],[1,-1,1,1]])
    out=inspect_solutions(tmp_path/'QP_TEST03.json',rows)
    assert out['raw_feasible_rate']==.5 and out['n_returned_rows']==2
    assert not out['physical_run_verified']
    assert out['rows'][0]['raw_conflicts']==2
    assert out['rows'][0]['classically_repaired_reward']==8
    file=tmp_path/'raw.csv';np.savetxt(file,rows,delimiter=',',fmt='%d')
    assert np.array_equal(read_solutions(file),rows)
    with pytest.raises(ValueError):inspect_solutions(tmp_path/'QP_TEST03.json',[[1,0,1,1]])

def test_strict_domain_checks_do_not_round_invalid_input():
    assert not preflight_ising(np.array([[0.,127.0001],[127.0001,0.]]))['valid']
    with pytest.raises(ValueError):repair(PackingInstance([1],[]),[.5])
    with pytest.raises(ValueError):safe_lns(PackingInstance([1],[]),[.5],[0])
    with pytest.raises(ValueError):PackingInstance([1,2],[(0,.5)])
