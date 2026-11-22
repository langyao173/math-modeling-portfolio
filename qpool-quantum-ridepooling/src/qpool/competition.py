"""Competition-oriented packing pipeline: exact reductions and 8-bit export.

The routines operate on a frozen one-period allocation objective. They do not
claim to preserve an entire closed-loop trajectory, nor to emulate hardware.
"""
from __future__ import annotations
from dataclasses import dataclass
import math,time
import numpy as np
from .qubo import (PackingInstance, induced, connected_components, packing_qubo,
                   qubo_to_ising, preflight_ising)
from .solvers import solve_milp, solve_greedy

@dataclass
class Reduction:
    instance: PackingInstance
    kept: list[int]
    fixed: list[int]
    removed: list[int]
    actions: list[dict]

    def lift(self, reduced_bits: np.ndarray, n: int) -> np.ndarray:
        x=np.asarray(reduced_bits)
        if x.shape!=(len(self.kept),) or not np.isin(x,[0,1]).all():
            raise ValueError('Reduced answer has the wrong shape or domain.')
        full=np.zeros(n,dtype=int);full[self.fixed]=1;full[self.kept]=x
        return full

def reduce_packing(inst: PackingInstance) -> Reduction:
    """Remove a dominated vertex v only when N[u] subset N[v] and w[u]>=w[v].

    N denotes the CLOSED active neighborhood. This is a standard weighted
    independent-set exchange rule. Positive isolated vertices are fixed to 1.
    Ties are resolved by index to keep the transformation deterministic.
    """
    n=len(inst.weights);adj=inst.adjacency();active=set(range(n));fixed=[];removed=[];actions=[]
    while True:
        changed=False
        for v in sorted(active):
            if inst.weights[v]<=0:
                active.remove(v);removed.append(v);actions.append({'rule':'nonpositive','v':v})
                changed=True;break
            neighbors=adj[v]&active
            if not neighbors:
                active.remove(v);fixed.append(v);actions.append({'rule':'isolated','v':v})
                changed=True;break
            Nv=neighbors|{v}
            for u in sorted(neighbors):
                if inst.weights[u]<inst.weights[v]:continue
                Nu=(adj[u]&active)|{u}
                if Nu<=Nv and (inst.weights[u]>inst.weights[v] or Nu!=Nv or u<v):
                    active.remove(v);removed.append(v)
                    actions.append({'rule':'closed_neighborhood_dominance','v':v,'witness':u})
                    changed=True;break
            if changed:break
        if not changed:break
    kept=sorted(active)
    return Reduction(induced(inst,kept),kept,sorted(fixed),sorted(removed),actions)

def compile_integer_packing(inst: PackingInstance, margin: int=1):
    """Exact rational-to-integer conversion, fast for integral gains/penalties."""
    if not np.array_equal(inst.weights,np.rint(inst.weights)):
        raise ValueError('Integer gains required; quantize explicitly first.')
    if not isinstance(margin,(int,np.integer)) or margin<=0:
        raise ValueError('The integer penalty margin must be positive.')
    Q=packing_qubo(inst,float(margin));J,bias=qubo_to_ising(Q)
    eight=J*8
    if not np.allclose(eight,np.rint(eight),rtol=0,atol=1e-10):
        raise ValueError('Unexpected non-integral scaled Ising coefficient.')
    I=np.rint(eight).astype(np.int64)
    g=int(np.gcd.reduce(np.abs(I).ravel())) or 1
    I//=g
    return Q,I,{'ising_scale':8/g,'qubo_bias':bias,'penalty_margin':margin,
                'energy_relation':'E_qubo = H_ising / ising_scale + qubo_bias',
                'ising_energy_convention':'H=-s.T@J@s','qubo_convention':'E=x.T@Q@x',
                'reference_spin':'last','preflight':preflight_ising(I)}

def choose_precision(inst: PackingInstance, resolutions=(32,16,8,4,2,1,.5,.25)):
    """Choose the finest passing precision, without observing any solver score."""
    attempts=[]
    for r in resolutions:
        if r<=0:raise ValueError('Precision resolutions must be positive.')
        quant=PackingInstance(np.rint(inst.weights*r),inst.edges,inst.labels)
        Q,I,meta=compile_integer_packing(quant)
        attempts.append({'resolution':r,**meta['preflight']})
        if meta['preflight']['valid']:
            meta.update({'resolution':r,'precision_attempts':attempts})
            return quant,Q,I,meta
    raise ValueError('No tested precision fits SPQC-550. Split/condition the problem; do not clip coefficients.')

def solve_competition_classical(inst: PackingInstance, time_limit: float=1.):
    """CPU-backed test of the QPool pipeline. This is NEVER quantum evidence.

    Reduction, compilation and decode are included in elapsed_s. If a component
    still exceeds the hardware model, a marked classical fallback is used.
    The original-utility greedy seed is accepted only if it improves the decoded
    candidate; this safeguard is separately logged.
    """
    started=time.perf_counter();red=reduce_packing(inst);bits=np.zeros(len(red.kept),int)
    kernels=[];fallbacks=0
    for ids in connected_components(red.instance):
        sub=induced(red.instance,ids)
        try:
            quant,Q,I,meta=choose_precision(sub)
            y,si=solve_milp(quant,max(.001,time_limit-(time.perf_counter()-started)))
            kernels.append({'binary_variables':len(ids),'n_spins':len(I),
                            'resolution':meta['resolution'],'optimal':si['optimal']})
        except ValueError:
            y,si=solve_milp(sub,max(.001,time_limit-(time.perf_counter()-started)));fallbacks+=1
            kernels.append({'binary_variables':len(ids),'fallback':'classical_uncompiled_component'})
        bits[ids]=y
    x=red.lift(bits,len(inst.weights))
    if not inst.evaluate(x)['feasible']:raise RuntimeError('Reduced solution lifted to an infeasible action.')
    seed,_=solve_greedy(inst);safeguard=float(inst.weights@seed)>float(inst.weights@x)+1e-10
    if safeguard:x=seed
    return x,{'elapsed_s':time.perf_counter()-started,'optimal':False,
              'backend':'QPool_CPU_compilation_pipeline_NOT_quantum',
              'fixed_variables':len(red.fixed),'dominated_variables':len(red.removed),
              'remaining_variables':len(red.kept),'n_kernels':len(kernels),
              'n_classical_fallbacks':fallbacks,'greedy_safeguard_used':safeguard,'kernels':kernels}

@dataclass
class HardwarePlan:
    original: PackingInstance
    core: PackingInstance
    quantized: PackingInstance
    kept: list[int]
    fixed: list[int]
    seed: np.ndarray
    Q: np.ndarray
    J: np.ndarray
    meta: dict

    def lift(self, bits):
        x=np.asarray(bits)
        if x.shape!=(len(self.kept),) or not np.isin(x,[0,1]).all():
            raise ValueError('Wrong core result size/domain.')
        full=np.zeros(len(self.original.weights),int);full[self.fixed]=1;full[self.kept]=x
        return full

def prepare_hardware_plan(inst: PackingInstance, seed=None, max_variables=96, min_resolution=1.):
    """Keep high-fidelity gains; bound a conditional neighborhood if necessary.

    Exact reduction is used first. When further truncation is required, omitted
    vertices are fixed to their seed values; all conflicts with fixed selections
    are removed. The resulting model is explicitly LOCAL, never called the full
    problem. Selection depends on gains/conflicts, not on any hardware results.
    """
    from .qubo import safe_lns
    if max_variables<2 or max_variables>549:raise ValueError('max_variables must be 2..549.')
    if seed is None:seed=solve_greedy(inst)[0]
    seed=np.asarray(seed,int)
    if not inst.evaluate(seed)['feasible']:raise ValueError('Seed must be feasible.')
    red=reduce_packing(inst)
    # When reduced core vanishes, the CPU already has a constructive optimum.
    if not red.kept:return None,{'fixed_bits':red.lift(np.array([],int),len(inst.weights)).tolist(),
                                'reason':'exact_reduction_solved'}
    allowed=tuple(r for r in (32,16,8,4,2,1,.5,.25) if r>=min_resolution)
    if not allowed:raise ValueError('Unsupported minimum precision.')
    if len(red.kept)<=max_variables:
        try:
            quant,Q,J,m=choose_precision(red.instance,allowed)
            m.update({'scope':'exact_reduced_full_period','original_n':len(inst.weights),
                      'fixed_original_utility':float(inst.weights[red.fixed].sum()),'reduction_actions':red.actions})
            return HardwarePlan(inst,red.instance,quant,red.kept,red.fixed,seed.copy(),Q,J,m),{}
        except ValueError:pass
    adj=inst.adjacency()
    # Expand around currently selected actions that face the most alternatives.
    anchors=sorted(range(len(inst.weights)),key=lambda i:(-len(adj[i]),-inst.weights[i],i))
    pool=[];seen=set()
    for a in anchors:
        for i in [a]+sorted(adj[a],key=lambda j:(-inst.weights[j],j)):
            if i not in seen:pool.append(i);seen.add(i)
    for cap in sorted({max_variables,min(max_variables,64),min(max_variables,48),min(max_variables,32),min(max_variables,16)},reverse=True):
        sub,kept,fixed=safe_lns(inst,seed,pool[:cap])
        kept=list(map(int,kept));fixed=list(map(int,fixed))
        if not kept:continue
        try:quant,Q,J,m=choose_precision(sub,allowed)
        except ValueError:continue
        m.update({'scope':'conditional_neighborhood','original_n':len(inst.weights),
                  'fixed_original_utility':float(inst.weights[fixed].sum()),
                  'omitted_variables_fixed_to_seed':True,'neighborhood_cap':cap})
        return HardwarePlan(inst,sub,quant,kept,fixed,seed.copy(),Q,J,m),{}
    return None,{'fixed_bits':seed.tolist(),'reason':'no_compilable_neighborhood_use_classical_seed'}

def solve_competition_lns_cpu(inst: PackingInstance, time_limit: float=1.):
    """Classical-backend rehearsal of a hardware-ready neighborhood + safeguard."""
    from .solvers import solve_local
    started=time.perf_counter()
    seed,_=solve_local(inst,min(.02,time_limit/4))
    plan,other=prepare_hardware_plan(inst,seed,max_variables=96,min_resolution=1.)
    if plan is None:
        x=np.asarray(other['fixed_bits'],int);n_kernels=0;kept=0
        accepted=False;scope=other['reason']
    else:
        y,si=solve_milp(plan.quantized,max(.001,time_limit-(time.perf_counter()-started)))
        y,polish=polish_equal_quantized_reward(plan.core,plan.quantized,y,min(.02,max(.001,time_limit-(time.perf_counter()-started))))
        proposed=plan.lift(y)
        if not inst.evaluate(proposed)['feasible']:raise RuntimeError('Lifted plan violates a conflict.')
        # Keep the classical seed on a tie; avoid pointless assignment churn.
        accepted=float(inst.weights@proposed)>float(inst.weights@seed)+1e-10
        x=proposed if accepted else seed
        n_kernels=1;kept=len(plan.kept);scope=plan.meta['scope']
    return x,{'elapsed_s':time.perf_counter()-started,'optimal':False,
              'backend':'QPool_LNS_CPU_rehearsal_NOT_quantum','n_kernels':n_kernels,
              'remaining_variables':kept,'n_classical_fallbacks':int(plan is None),
              'greedy_safeguard_used':not accepted,'scope':scope}

def polish_equal_quantized_reward(original: PackingInstance, quantized: PackingInstance, bits,
                                   time_limit: float=.02):
    """Classical tie breaking at unchanged quantized reward, reported separately.

    One-/two-candidate exchanges recover original-utility distinctions lost in
    quantization. Every accepted move stays feasible and leaves the submitted
    quantized objective unchanged. Apply the same routine to every backend.
    """
    started=time.perf_counter();x=np.asarray(bits,int).copy()
    if original.edges!=quantized.edges or len(original.weights)!=len(quantized.weights):
        raise ValueError('Original and quantized instances must share a feasible domain.')
    if not original.evaluate(x)['feasible']:raise ValueError('Tie polishing requires a feasible input.')
    adj=original.adjacency();w=original.weights;q=quantized.weights
    before=float(w@x);target=float(q@x);moves=0
    while time.perf_counter()-started<time_limit:
        inactive=list(map(int,np.flatnonzero(1-x)));best=None;best_gain=1e-10
        for k,u in enumerate(inactive):
            victims=[j for j in adj[u] if x[j]]
            if abs(q[u]-sum(q[j] for j in victims))<1e-10:
                gain=w[u]-sum(w[j] for j in victims)
                if gain>best_gain:best_gain=gain;best=([u],victims)
            for v in inactive[k+1:]:
                if time.perf_counter()-started>=time_limit:break
                if v in adj[u]:continue
                victims2=[j for j in adj[u]|adj[v] if x[j]]
                if abs(q[u]+q[v]-sum(q[j] for j in victims2))<1e-10:
                    gain=w[u]+w[v]-sum(w[j] for j in victims2)
                    if gain>best_gain:best_gain=gain;best=([u,v],victims2)
            if time.perf_counter()-started>=time_limit:break
        if best is None:break
        add,drop=best;x[drop]=0;x[add]=1;moves+=1
    if not original.evaluate(x)['feasible'] or not np.isclose(q@x,target):
        raise RuntimeError('Tie polish broke its invariants.')
    return x,{'classical_postprocessing':True,'elapsed_s':time.perf_counter()-started,
              'n_exchanges':moves,'raw_original_utility':before,'polished_original_utility':float(w@x),
              'quantized_reward_unchanged':True,'quantized_reward':target}
