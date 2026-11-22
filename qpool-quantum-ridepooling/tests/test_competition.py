import numpy as np
import pytest
from qpool.qubo import PackingInstance,exact_enumeration,compile_instance,decode_spins
from qpool.competition import reduce_packing,compile_integer_packing,choose_precision,solve_competition_classical

@pytest.mark.parametrize('seed',range(40))
def test_reduction_preserves_optimum(seed):
    r=np.random.default_rng(seed);n=9
    edges=[(i,j) for i in range(n) for j in range(i+1,n) if r.random()<.4]
    inst=PackingInstance(r.integers(0,12,n),edges)
    red=reduce_packing(inst);v,x=exact_enumeration(inst)
    if not red.kept:lift=red.lift(np.array([],int),n)
    else:
        _,small=exact_enumeration(red.instance);lift=red.lift(small[0],n)
    assert inst.evaluate(lift)['feasible']
    assert float(inst.weights@lift)==v

@pytest.mark.parametrize('seed',range(10))
def test_fast_integer_compiler_matches_reference(seed):
    r=np.random.default_rng(seed);inst=PackingInstance(r.integers(1,9,8),[(i,j) for i in range(8) for j in range(i+1,8) if r.random()<.4])
    Q,I,m=compile_integer_packing(inst);Q2,I2,m2=compile_instance(inst)
    assert np.array_equal(Q,Q2) and np.array_equal(I,I2)
    assert m['ising_scale']==m2['ising_scale']
    for _ in range(20):
        s=2*r.integers(0,2,9)-1;x=decode_spins(s,8)
        assert np.isclose(x@Q@x,(-s@I@s)/m['ising_scale']+m['qubo_bias'])

def test_precision_gate_and_cpu_label():
    inst=PackingInstance([1.17,2.81,1.03,2.19,1.07],[(0,1),(1,2),(2,3),(3,4),(4,0)])
    quant,Q,J,m=choose_precision(inst)
    assert m['preflight']['valid'] and all(a['valid']==False for a in m['precision_attempts'][:-1])
    x,info=solve_competition_classical(inst)
    assert inst.evaluate(x)['feasible'] and 'NOT_quantum' in info['backend']

@pytest.mark.parametrize('seed',range(12))
def test_conditioned_core_lifts_feasibly_and_contains_seed(seed):
    from qpool.competition import prepare_hardware_plan
    from qpool.solvers import solve_greedy,solve_milp
    rng=np.random.default_rng(seed);n=35
    inst=PackingInstance(rng.uniform(.7,3,n),[(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<.2])
    seed_x=solve_greedy(inst)[0]
    p,other=prepare_hardware_plan(inst,seed_x,max_variables=16)
    if p is None:
        assert inst.evaluate(np.asarray(other['fixed_bits']))['feasible'];return
    x,_=solve_milp(p.quantized,3);full=p.lift(x)
    assert inst.evaluate(full)['feasible'] and p.meta['preflight']['valid']
    assert p.meta['resolution']>=1
    if p.meta['scope']=='conditional_neighborhood':
        assert np.array_equal(p.lift(seed_x[p.kept]),seed_x)

@pytest.mark.parametrize('seed',range(10))
def test_tie_polish_keeps_quantized_energy_and_feasibility(seed):
    from qpool.competition import polish_equal_quantized_reward
    from qpool.solvers import solve_milp
    rng=np.random.default_rng(seed);n=12
    inst=PackingInstance(rng.uniform(.7,3,n),[(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<.3])
    quant=PackingInstance(np.rint(inst.weights),inst.edges)
    x,_=solve_milp(quant,2);y,info=polish_equal_quantized_reward(inst,quant,x,.2)
    assert inst.evaluate(y)['feasible'] and quant.weights@y==quant.weights@x
    assert inst.weights@y>=inst.weights@x-1e-10

@pytest.mark.parametrize('name',[f'QP_A{i:02d}' for i in range(1,7)])
def test_application_batch_reference_answers(name):
    from pathlib import Path
    import json
    from qpool.results import inspect_solutions
    path=Path(__file__).resolve().parents[1]/'upload/03_application_batch'/f'{name}.json'
    m=json.loads(path.read_text());x=np.array(m['exact_quantized_bits'],int)
    s=np.r_[2*x-1,1]
    report=inspect_solutions(path,np.array([s]),source='unit_test_known_classical_reference')
    row=report['rows'][0]
    assert row['ising_energy']==m['exact_ising_minimum']
    assert row['raw_feasible'] and row['lifted_raw_feasible']
    assert m['preflight']['valid'] and m['resolution']>=1
