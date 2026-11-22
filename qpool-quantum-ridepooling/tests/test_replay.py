import numpy as np
from qpool.replay import *

def test_locked_leg_cannot_be_teleported():
    tt=np.array([[0,100,200],[100,0,100],[200,100,0]])
    c=Config(capacity=2,dwell=10,max_wait=1000,max_detour=500)
    rs={0:Request(0,1,2,0,status='accepted',vehicle=0),1:Request(1,0,2,0)}
    v=Vehicle(0,0,route=[Stop(0,1,True),Stop(0,2,False)])
    start_leg(v,0,tt);advance(v,50,rs,tt,c)
    out=insert_one(v,v.route,rs[1],50,rs,tt,c)
    assert out is not None and out[0][0]==Stop(0,1,True)
    original_eta=v.eta
    ca=Candidate(0,(1,),out[0],1,np.zeros(3),0)
    apply([v],[ca],[1],50,rs,tt,c)
    assert v.eta==original_eta==100
    advance(v,1000,rs,tt,c)
    assert all(r.status=='completed' for r in rs.values()) and v.driving==400
    assert all(r.pickup>=r.arrival for r in rs.values())


def test_empty_selection_preserves_commitments():
    tt=np.array([[0,10],[10,0]])
    rs={0:Request(0,0,1,0,status='accepted',vehicle=0)}
    c=Config(dwell=0);v=Vehicle(0,0,route=[Stop(0,0,True),Stop(0,1,False)])
    start_leg(v,0,tt);apply([v],[],[],0,rs,tt,c);advance(v,100,rs,tt,c)
    assert rs[0].status=='completed' and rs[0].vehicle==0


def test_wait_and_capacity_are_not_soft_penalties():
    tt=np.array([[0,100],[100,0]])
    c=Config(max_wait=50);rs={0:Request(0,1,0,0)};v=Vehicle(0,0)
    assert insert_one(v,[],rs[0],0,rs,tt,c) is None
    c=Config(capacity=0)
    assert insert_one(v,[],rs[0],0,rs,tt,c) is None

def test_stale_result_rejected_atomically():
    import pytest
    tt=np.array([[0,100],[100,0]]);cfg=Config(max_wait=600,dwell=0)
    rs={0:Request(0,0,1,0),1:Request(1,0,1,0)}
    vs=[Vehicle(0,0),Vehicle(1,0)]
    cs=[Candidate(0,(0,),[Stop(0,0,True),Stop(0,1,False)],1,np.zeros(2),100),
        Candidate(1,(1,),[Stop(1,0,True),Stop(1,1,False)],1,np.zeros(2),100)]
    rs[1].status='rejected'
    with pytest.raises(ValueError):apply(vs,cs,[1,1],0,rs,tt,cfg)
    assert rs[0].status=='pending' and not vs[0].route and not vs[1].route
