from pathlib import Path
import copy
import json
import numpy as np
import pytest
from qpool.qubo import PackingInstance
from qpool.provider_reports import (read_provider_pdf, parse_report_text, verify_provider_report,
                                    select_raw_candidate, dispatch_gate)

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'results/stage3/provider_reports'
MODELS=ROOT/'upload/03_application_batch'
ENERGIES=[-1250,-3750,-490,-3564,-3534,-4512]
FEASIBLE=[10,3,1,7,9,7]
OPTIMAL=[10,1,1,1,2,5]
SPANS=[5,6,4,1609,1603,1602]

@pytest.fixture(scope='module')
def checked():
    return [verify_provider_report(REPORTS/f'QP_APP_A_{i:02d}.pdf', MODELS) for i in range(1,7)]

@pytest.mark.parametrize('i',range(6))
def test_every_provider_report_is_consistent(checked,i):
    r=checked[i];s=r['summary']
    assert len(r['rows'])==10 and s['best_raw_H']==ENERGIES[i]
    assert s['absolute_H_gap']==0 and s['raw_feasible_rows']==FEASIBLE[i]
    assert s['optimal_rows']==OPTIMAL[i] and s['created_to_completed_s']==SPANS[i]
    assert r['rows'][0]['raw_feasible']
    assert all(row['energy_residual']==0 for row in r['rows'])
    assert not r['input_matrix_embedded_in_report']
    assert not s['is_full_day_hardware_replay']

@pytest.mark.parametrize('i',range(6))
def test_reference_spin_global_flip_preserves_every_hardware_row(checked,i):
    from qpool.qubo import decode_spins
    for row in checked[i]['rows']:
        spins=np.array(row['spins']);n=len(spins)-1
        assert np.array_equal(decode_spins(spins,n),decode_spins(-spins,n))

@pytest.mark.parametrize('i',range(6))
def test_raw_output_remains_separate_from_polish(checked,i):
    r=checked[i]
    for row in r['rows']:
        if row['raw_feasible']:
            assert row['tie_polished_full_original_utility'] >= row['lifted_raw_original_utility']-1e-10
            assert row['tie_polish']['quantized_reward_unchanged']
        else:
            assert 'tie_polish' not in row
    assert r['provider']['sampling_count_as_reported']=='N/A'
    assert r['rows_are_not_assumed_independent_runs']


def test_returned_pool_selects_original_utility_tie(checked):
    raw=copy.deepcopy(checked[0]['rows'])
    best=select_raw_candidate(raw)
    assert best['result_index']==7
    assert best['lifted_raw_original_utility']==pytest.approx(2.88)
    assert raw==checked[0]['rows']
    assert select_raw_candidate([]) is None

@pytest.mark.parametrize('replacement', ['结果2 ： Hamiltonian', '结果3 ： Hamiltonian'])
def test_bad_result_numbering_fails(replacement):
    import fitz
    with fitz.open(REPORTS/'QP_APP_A_01.pdf') as doc:text='\n'.join(p.get_text() for p in doc)
    text=text.replace('结果1 ： Hamiltonian', replacement,1)
    with pytest.raises(ValueError):parse_report_text(text)


def test_incomplete_vector_fails():
    import fitz
    with fitz.open(REPORTS/'QP_APP_A_01.pdf') as doc:text='\n'.join(p.get_text() for p in doc)
    with pytest.raises(ValueError):parse_report_text(text.replace(']','',1))


def test_invalid_spin_token_fails():
    import fitz
    with fitz.open(REPORTS/'QP_APP_A_01.pdf') as doc:text='\n'.join(p.get_text() for p in doc)
    with pytest.raises(ValueError):parse_report_text(text.replace('[-1,','[0,',1))

@pytest.mark.parametrize('failure',['energy','dimension','machine','type','status'])
def test_report_verifier_rejects_corruption(monkeypatch,failure):
    from qpool import provider_reports as p
    r=read_provider_pdf(REPORTS/'QP_APP_A_01.pdf')
    if failure=='energy':r['rows'][0]['reported_H']+=1
    elif failure=='dimension':r['rows'][0]['spins'].pop()
    elif failure=='machine':r['machine']='SPQC-1000'
    elif failure=='type':r['matrix_type']='QUBO'
    elif failure=='status':r['status']='失败'
    monkeypatch.setattr(p,'read_provider_pdf',lambda path:r)
    with pytest.raises(ValueError):p.verify_provider_report(REPORTS/'QP_APP_A_01.pdf',MODELS)

@pytest.mark.parametrize('elapsed,submitted,current,proposed,reason',[
    (61,'a','a',[0,1],'deadline_exceeded'),
    (1,'a','b',[0,1],'state_changed'),
    (1,'a','a',None,'no_feasible_returned_candidate'),
    (1,'a','a',[1,1],'infeasible_returned_candidate'),
    (1,'a','a',[1,0],'no_strict_original_utility_improvement'),
    (1,'a','a',[0,1],'strict_feasible_improvement'),
])
def test_dispatch_gate(elapsed,submitted,current,proposed,reason):
    inst=PackingInstance([1.,2.],[(0,1)]);seed=np.array([1,0])
    out,info=dispatch_gate(inst,seed,proposed,submitted_state=submitted,current_state=current,elapsed_s=elapsed,deadline_s=60)
    assert info['reason']==reason
    assert inst.evaluate(out)['feasible']
    assert np.array_equal(out,[0,1] if info['accepted'] else [1,0])

@pytest.mark.parametrize('elapsed,deadline',[(-1,60),(0,0),(float('inf'),60),(1,float('nan'))])
def test_gate_invalid_clocks_fail(elapsed,deadline):
    with pytest.raises(ValueError):dispatch_gate(PackingInstance([1],[]),[0],[1],submitted_state='x',current_state='x',elapsed_s=elapsed,deadline_s=deadline)
