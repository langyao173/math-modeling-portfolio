"""Independent commitment-preserving insertion replay, not an official RTV reproduction.

A selected variable inserts one or more *new* requests into one vehicle's existing
route. No variable selected means continue the incumbent route. The already
started leg is locked. Every candidate preserves all previously accepted stops.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import csv,json,time,math
import numpy as np
from .qubo import PackingInstance, induced
from .solvers import solve_greedy,solve_local,solve_milp,solve_joint_tv,tv_worst_probability

@dataclass
class Config:
    fleet:int=8
    capacity:int=4
    epoch:int=60
    max_wait:int=600
    max_detour:int=600
    dwell:int=60
    shortlist:int=10
    beam:int=10
    max_new_requests:int=3
    seed:int=11
    solver_budget:float=1.0
    cost_weight:float=.08
    future_weight:float=1.0
    tv_radius:float=.10
    future_horizon:int=1800

@dataclass
class Request:
    rid:int
    origin:int
    destination:int
    arrival:int
    status:str='pending'
    vehicle:int=-1
    pickup:float|None=None
    dropoff:float|None=None

@dataclass(frozen=True)
class Stop:
    rid:int
    node:int
    pickup:bool

@dataclass
class Vehicle:
    vid:int
    node:int
    ready:float=0.
    route:list[Stop]=field(default_factory=list)
    eta:float|None=None
    leg_depart:float=0.
    accounted_until:float=0.
    onboard:set[int]=field(default_factory=set)
    driving:float=0.
    empty_driving:float=0.

@dataclass
class Candidate:
    vehicle:int
    requests:tuple[int,...]
    route:list[Stop]
    reward:float
    feature:np.ndarray
    remaining_drive:float


def load_data(folder:Path):
    folder=Path(folder)
    rows=list(csv.DictReader((folder/'SiouxFalls_TTmatrix.csv').open()))
    n=max(max(int(r['origin']),int(r['destination'])) for r in rows)+1
    tt=np.full((n,n),np.nan)
    for r in rows:tt[int(r['origin']),int(r['destination'])]=float(r['travel_time'])
    if not np.isfinite(tt).all() or np.any(tt<0):raise ValueError('Travel-time matrix incomplete/invalid.')
    rs=[Request(int(r['request_id']),int(r['origin']),int(r['destination']),int(r['arrival']))
        for r in csv.DictReader((folder/'requests.csv').open())]
    if len({r.rid for r in rs})!=len(rs):raise ValueError('Duplicate request ID.')
    if any(not 0<=r.origin<n or not 0<=r.destination<n for r in rs):raise ValueError('Invalid node ID.')
    return tt,sorted(rs,key=lambda r:(r.arrival,r.rid))


def start_leg(v:Vehicle,now:float,tt):
    if v.route:
        v.leg_depart=max(now,v.ready)
        v.eta=v.leg_depart+tt[v.node,v.route[0].node]
        v.accounted_until=v.leg_depart
    else:v.eta=None


def account_leg(v:Vehicle,until:float):
    if v.eta is None:return
    end=min(until,v.eta);begin=max(v.accounted_until,v.leg_depart)
    d=max(0.,end-begin);v.driving+=d
    if not v.onboard:v.empty_driving+=d
    v.accounted_until=max(v.accounted_until,end)


def advance(v:Vehicle,now:float,reqs:dict,tt,cfg:Config):
    while v.route and v.eta is not None and v.eta<=now+1e-9:
        account_leg(v,v.eta);s=v.route.pop(0);at=float(v.eta);v.node=s.node
        r=reqs[s.rid]
        if s.pickup:
            assert r.pickup is None and r.status=='accepted' and r.vehicle==v.vid
            assert at-r.arrival<=cfg.max_wait+1e-8
            r.pickup=at;r.status='onboard';v.onboard.add(r.rid)
            assert len(v.onboard)<=cfg.capacity
        else:
            assert s.rid in v.onboard and r.pickup is not None
            assert at-r.pickup<=tt[r.origin,r.destination]+cfg.max_detour+1e-8
            v.onboard.remove(s.rid);r.dropoff=at;r.status='completed'
        v.ready=at+cfg.dwell;start_leg(v,v.ready,tt)
    account_leg(v,now)


def evaluate_route(v,route,now,reqs,tt,cfg):
    """Exact validation for the represented ordered route, not global routing optimization."""
    if v.route and (not route or route[0]!=v.route[0]):return None
    # Candidate generation preserves incumbent order; the caller also checks it on commit.
    onboard=set(v.onboard);pickup={rid:reqs[rid].pickup for rid in onboard}
    node=v.node;at=max(now,v.ready);drive=0.
    for k,s in enumerate(route):
        if k==0 and v.route:
            if v.eta is None:raise AssertionError('Missing locked-leg ETA.')
            travel=max(0.,v.eta-max(now,v.leg_depart));at=v.eta
        else:travel=tt[node,s.node];at+=travel
        drive+=travel;r=reqs[s.rid]
        if s.pickup:
            if s.rid in onboard or s.rid in pickup or at-r.arrival>cfg.max_wait+1e-8:return None
            pickup[s.rid]=at;onboard.add(s.rid)
            if len(onboard)>cfg.capacity:return None
        else:
            if s.rid not in onboard or s.rid not in pickup:return None
            if at-pickup[s.rid]>tt[r.origin,r.destination]+cfg.max_detour+1e-8:return None
            onboard.remove(s.rid)
        node=s.node;at+=cfg.dwell
    if onboard:return None
    return drive,node,at


def insert_one(v,route,r,now,reqs,tt,cfg):
    start=1 if v.route else 0;best=None;best_key=None
    for i in range(start,len(route)+1):
        r1=route[:i]+[Stop(r.rid,r.origin,True)]+route[i:]
        for j in range(i+1,len(r1)+1):
            r2=r1[:j]+[Stop(r.rid,r.destination,False)]+r1[j:]
            score=evaluate_route(v,r2,now,reqs,tt,cfg)
            if score is None:continue
            key=(score[0],score[2],tuple((s.rid,s.pickup) for s in r2))
            if best_key is None or key<best_key:best=(r2,score);best_key=key
    return best


def availability(node,ready,now,tt,cfg):
    return np.exp(-(max(0.,ready-now)+tt[node,:])/cfg.future_horizon)


def candidates(vehicles,pending,now,reqs,tt,cfg):
    allc=[]
    for v in vehicles:
        base=evaluate_route(v,v.route,now,reqs,tt,cfg)
        assert base is not None
        base_drive,base_node,base_ready=base
        seen={};frontier=[]
        # Enumerate every feasible singleton before truncating the extension beam.
        # A nearer but expired/infeasible request must not block a feasible request.
        single=[]
        for r in pending:
            out=insert_one(v,v.route,r,now,reqs,tt,cfg)
            if out:single.append(((r.rid,),*out))
        frontier=single
        extension_ids={ids[0] for ids,_,score in sorted(single,key=lambda d:(d[2][0],d[0]))[:cfg.shortlist]}
        shortlist=[reqs[rid] for rid in sorted(extension_ids)]
        for depth in range(1,cfg.max_new_requests+1):
            for ids,route,score in frontier:
                if ids not in seen or score[0]<seen[ids][1][0]:seen[ids]=(route,score)
            if depth==cfg.max_new_requests:break
            nxt={}
            for ids,route,_ in sorted(frontier,key=lambda d:(d[2][0],d[0]))[:cfg.beam]:
                for r in shortlist:
                    if r.rid in ids:continue
                    ids2=tuple(sorted((*ids,r.rid)));out=insert_one(v,route,r,now,reqs,tt,cfg)
                    if out and (ids2 not in nxt or out[1][0]<nxt[ids2][1][0]):nxt[ids2]=out
            frontier=[(ids,*out) for ids,out in sorted(nxt.items())]
        for ids,(route,score) in sorted(seen.items()):
            drive,node,ready=score
            gain=len(ids)-cfg.cost_weight*(drive-base_drive)/600.
            if gain<=0:continue
            feat=availability(node,ready,now,tt,cfg)-availability(base_node,base_ready,now,tt,cfg)
            allc.append(Candidate(v.vid,ids,route,float(gain),feat,float(drive)))
    labels=[{'vehicle':c.vehicle,'requests':list(c.requests)} for c in allc]
    groups={};es=set()
    from itertools import combinations
    for i,d in enumerate(labels):
        for k in [('v',d['vehicle'])]+[('r',r) for r in d['requests']]:groups.setdefault(k,[]).append(i)
    for g in groups.values():es.update(combinations(g,2))
    inst=PackingInstance(np.array([c.reward for c in allc]),list(es),labels)
    features=np.array([c.feature for c in allc]).T if allc else np.zeros((len(tt),0))
    return allc,inst,features


def apply(vehicles,cs,x,now,reqs,tt,cfg):
    """Validate the entire action before committing any mutation.

    Stale external answers are rejected atomically; the caller can then select a
    fresh classical fallback. Validation is not disabled by Python's -O option.
    """
    x=np.asarray(x)
    if x.shape!=(len(cs),) or not np.isin(x,[0,1]).all():raise ValueError('Invalid action vector.')
    vmap={v.vid:v for v in vehicles};chosen=[cs[i] for i in np.flatnonzero(x)]
    used_v=set();used_r=set()
    for c in chosen:
        if c.vehicle not in vmap or c.vehicle in used_v:raise ValueError('Repeated/unknown vehicle in action.')
        if used_r.intersection(c.requests):raise ValueError('Repeated request in action.')
        used_v.add(c.vehicle);used_r.update(c.requests);v=vmap[c.vehicle]
        old=[stop for stop in c.route if stop.rid not in c.requests]
        if old!=v.route:raise ValueError('Stale action would change an incumbent commitment.')
        if evaluate_route(v,c.route,now,reqs,tt,cfg) is None:raise ValueError('Action no longer satisfies current route constraints.')
        for rid in c.requests:
            if rid not in reqs or reqs[rid].status!='pending' or reqs[rid].arrival>now:
                raise ValueError('Request is no longer available at commit time.')
    for c in chosen:
        v=vmap[c.vehicle];already_moving=bool(v.route)
        for rid in c.requests:
            r=reqs[rid];r.status='accepted';r.vehicle=v.vid
        v.route=list(c.route)
        if not already_moving:start_leg(v,now,tt)


def run_replay(data_folder,out,method='myopic_mip',cfg=None,capture=False):
    cfg=cfg or Config();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tt,arrivals=load_data(Path(data_folder));reqs={r.rid:r for r in arrivals}
    rng=np.random.default_rng(cfg.seed)
    vehicles=[Vehicle(i,int(rng.integers(len(tt)))) for i in range(cfg.fleet)]
    st=time.perf_counter();pointer=0;epoch_rows=[];captures=[];counts=np.ones(len(tt))
    last=max(r.arrival for r in arrivals);last_release=math.ceil(last/cfg.epoch)*cfg.epoch
    # Continue until every accepted passenger is delivered and pending requests expire.
    end=last_release+cfg.max_wait+10000
    for now in range(0,end+cfg.epoch,cfg.epoch):
        for v in vehicles:advance(v,now,reqs,tt,cfg)
        while pointer<len(arrivals) and arrivals[pointer].arrival<=now:
            counts[arrivals[pointer].origin]+=1;pointer+=1
        for r in arrivals[:pointer]:
            if r.status=='pending' and now-r.arrival>cfg.max_wait:r.status='rejected'
        pending=[r for r in arrivals[:pointer] if r.status=='pending']
        if pending:
            tick=time.perf_counter()
            effective_cfg=(replace(cfg,cost_weight=cfg.cost_weight*len(pending)/cfg.fleet) if method=='adaptive_price_mip' else cfg)
            cs,inst,Z=candidates(vehicles,pending,now,reqs,tt,effective_cfg)
            generate=time.perf_counter()-tick;p=counts/counts.sum()
            if method=='greedy':x,info=solve_greedy(inst)
            elif method=='local_search':x,info=solve_local(inst,cfg.solver_budget)
            elif method in ('myopic_mip','adaptive_price_mip'):x,info=solve_milp(inst,cfg.solver_budget)
            elif method=='qpool_lns_cpu':
                from .competition import solve_competition_lns_cpu
                x,info=solve_competition_lns_cpu(inst,cfg.solver_budget)
            elif method=='qpool_cpu':
                from .competition import solve_competition_classical
                x,info=solve_competition_classical(inst,cfg.solver_budget)
            elif method in ('point_mip','separable_tv_mip'):
                f=(p@Z if method=='point_mip' else np.array([tv_worst_probability(p,Z[:,i],cfg.tv_radius)[1] for i in range(Z.shape[1])]))
                raw_w=inst.weights+cfg.future_weight*f
                w=np.maximum(0,raw_w);keep=np.flatnonzero(raw_w>=0).tolist()
                altered=PackingInstance(w,inst.edges,inst.labels)
                sx,info=solve_milp(induced(altered,keep),cfg.solver_budget);x=np.zeros(len(w),int);x[keep]=sx
            elif method=='joint_tv_mip':x,info=solve_joint_tv(inst,Z,p,cfg.tv_radius,cfg.future_weight,cfg.solver_budget)
            else:raise ValueError('Unknown method '+method)
            assert inst.evaluate(x)['feasible']
            apply(vehicles,cs,x,now,reqs,tt,cfg)
            epoch_rows.append({'time':now,'pending':len(pending),'n_candidates':len(cs),
                               'edges':len(inst.edges),'selected':int(x.sum()),'new_accepts':sum(len(cs[i].requests) for i in np.flatnonzero(x)),
                               'reward':float(inst.weights@x),'generation_s':generate,'solve_s':info['elapsed_s'],
                               'decision_s':time.perf_counter()-tick,'optimal':info.get('optimal',False),
                               'fixed_variables':info.get('fixed_variables',0),'dominated_variables':info.get('dominated_variables',0),
                               'hardware_kernels':info.get('n_kernels',0),'classical_fallbacks':info.get('n_classical_fallbacks',0),
                               'greedy_safeguard_used':info.get('greedy_safeguard_used',False)})
            if capture and cs:
                # Snapshot before selection: preserves the baseline state's conditional decision problem.
                snap=inst.to_dict();snap.update({'epoch':now,'incumbent':x.tolist(),'features':Z.tolist(),'forecast':p.tolist(),
                    'candidate_routes':[[asdict(s) for s in c.route] for c in cs],
                    'source':'external Sioux Falls example prefix, independent replay',
                    'interpretation':'conditional snapshot; not a quantum closed-loop trajectory'})
                captures.append(snap)
        if pointer==len(arrivals) and not any(r.status in ('pending','accepted','onboard') for r in arrivals):break
    if any(r.status in ('pending','accepted','onboard') for r in arrivals):raise RuntimeError('Drain horizon too short.')
    completed=[r for r in arrivals if r.status=='completed'];rejected=[r for r in arrivals if r.status=='rejected']
    assert len(completed)+len(rejected)==len(arrivals)
    waits=np.array([r.pickup-r.arrival for r in completed]);rides=np.array([r.dropoff-r.pickup for r in completed])
    detours=np.array([r.dropoff-r.pickup-tt[r.origin,r.destination] for r in completed]);dt=np.array([e['decision_s'] for e in epoch_rows])
    def mean(a):return None if len(a)==0 else float(np.mean(a))
    def quant(a,p):return None if len(a)==0 else float(np.quantile(a,p))
    metrics={'method':method,'config':asdict(cfg),'n_requests':len(arrivals),'completed':len(completed),'rejected':len(rejected),
             'service_rate':len(completed)/len(arrivals),'mean_wait_s':mean(waits),'p95_wait_s':quant(waits,.95),
             'mean_invehicle_s':mean(rides),'mean_detour_s':mean(detours),'max_detour_s':quant(detours,1),
             'vehicle_driving_s':sum(v.driving for v in vehicles),'empty_driving_s':sum(v.empty_driving for v in vehicles),
             'decision_mean_s':mean(dt),'decision_p95_s':quant(dt,.95),'wall_clock_s':time.perf_counter()-st,
             'capacity_violations':0,'waiting_violations':0,'detour_violations':0,'unfinished':0,
             'max_candidates':max((e['n_candidates'] for e in epoch_rows),default=0),'evaluation_level':'EXTERNAL_EXAMPLE_DIAGNOSTIC_NOT_NATIVE_SOTA'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (out/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
    for name,rows in [('requests',[asdict(r) for r in arrivals]),('epochs',epoch_rows)]:
        if rows:
            with (out/(name+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if capture:(out/'snapshots.json').write_text(json.dumps(captures))
    return metrics
