"""Independently join frozen chemistry, real CSV outputs, and transfer cases.

Read-only result audit. No inference, fitting, selection, or source mutation.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MOD = ROOT / "experiments/modern_prediction"
CHALLENGE = MOD / "deployment_transfer_challenge"
FINAL = ROOT / "experiments/sparse_transfer/final_summary"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_root():
    for ancestor in ROOT.parents:
        for p in (ancestor, ancestor / "project/MathorCup_A", ROOT / "project/MathorCup_A"):
            if (p / "data/A/1.csv").exists():
                return p
    raise FileNotFoundError("Restored MathorCup_A project not found")


def main():
    # Fresh-run adaptation: preserve all identity/mapping checks;
    # compare independent counts with this run, not a previous solver's pool.
    assert (ROOT / 'COLD_RECOMPUTATION.json').is_file()
    complete = json.loads((CHALLENGE / "COMPLETE.json").read_text())
    assert complete["complete"] and complete["original_fstar_pool_unchanged"]
    before = {name:sha(FINAL/name) for name in complete["source_summary_sha256"]}
    assert before == complete["source_summary_sha256"]
    assert sha(FINAL / "selection_cases.csv") == complete["candidate_pool_sha256"]
    for name, expected in complete["result_sha256"].items():
        assert sha(CHALLENGE / name) == expected
    data = pd.read_csv(project_root() / "data/A/1.csv").set_index("beer")
    chemistry = pd.read_csv(CHALLENGE / "source_and_candidate_chemistry.csv").set_index("beer")
    predictions = pd.read_csv(CHALLENGE / "all50_sensory_predictions.csv").set_index("beer")
    selected = pd.read_csv(FINAL / "selected_profiles.csv").set_index(["evaluation_policy", "candidate_uid"])
    selection = pd.read_csv(FINAL / "selection_cases.csv")
    results = pd.read_csv(CHALLENGE / "cross_model_case_checks.csv")
    joined = selection.merge(results, on=["policy","beer","rho"], validate="one_to_one", suffixes=("_frozen","_challenge"))
    assert len(joined) == 88 and len(chemistry) == len(predictions) == complete['input_rows']
    assert len(chemistry) == complete['source_count'] + complete['unique_candidate_profiles']
    assert complete['source_count'] == 11
    assert chemistry.index.is_unique and chemistry.index.equals(predictions.index)
    assert np.isfinite(predictions.to_numpy()).all()
    assert (joined.candidate_uid_frozen == joined.candidate_uid_challenge).all()
    assert joined.accepted.all() and joined.fstar_full_accepted.all()
    assert np.max(np.abs(joined.ensemble_stale_delta-joined.fstar_stale_delta)) < 1e-14
    features = chemistry.columns.tolist()
    assert len(features) == 231
    max_chemical_error = 0.
    for beer in sorted(joined.beer.unique()):
        actual = chemistry.loc[f"source_{beer}",features].to_numpy(float)
        expected = data.loc[beer,features].to_numpy(float)
        assert np.array_equal(np.isnan(actual),np.isnan(expected))
        max_chemical_error = max(max_chemical_error,float(np.nanmax(np.abs(actual-expected))))
    rebuilt = []
    for row in joined.itertuples():
        actual = chemistry.loc[row.candidate_input_id,features].to_numpy(float)
        expected = selected.loc[(row.policy,row.candidate_uid_frozen),features].to_numpy(float)
        assert np.array_equal(np.isnan(actual),np.isnan(expected))
        max_chemical_error = max(max_chemical_error,float(np.nanmax(np.abs(actual-expected))))
        delta = float(predictions.loc[row.candidate_input_id,"stale_hops"]-predictions.loc[row.source_input_id,"stale_hops"])
        assert (delta <= 0) == row.fplus_nonincrease_exact
        assert (delta <= 1e-8) == row.fplus_nonincrease_at_1e_8
        rebuilt.append(dict(policy=row.policy, beer=row.beer, rho=row.rho,
            candidate_input_id=row.candidate_input_id, delta_recomputed=delta,
            delta_stored=row.fplus_stale_delta, residual=delta-row.fplus_stale_delta,
            nonincrease=delta <= 0))
    checks = pd.DataFrame(rebuilt)
    assert max_chemical_error < 1e-12
    assert checks.residual.abs().max() < 1e-14
    unique = checks.drop_duplicates(["beer","candidate_input_id"])
    assert len(unique) == complete['unique_source_candidate_pairs']
    assert int(unique.nonincrease.sum()) == complete['unique_pair_nonincrease_exact']
    assert int((~unique.nonincrease).sum()) == complete['unique_pair_increase']
    candidates = chemistry.loc[checks.candidate_input_id.unique(),features].to_numpy(float)
    assert len(np.unique(np.nan_to_num(candidates,nan=np.inf),axis=0)) == complete['unique_candidate_profiles']
    groups = checks.groupby(["policy","rho"],sort=False).agg(cases=("beer","size"),nonincrease=("nonincrease","sum"),delta_mean=("delta_recomputed","mean"))
    recorded = pd.read_csv(CHALLENGE / "policy_rho_summary.csv").set_index(["policy","rho"])
    for key, group in groups.iterrows():
        assert int(group.cases) == int(recorded.loc[key,"cases"])
        assert int(group.nonincrease) == int(recorded.loc[key,"fplus_nonincrease_exact"])
        assert abs(group.delta_mean-recorded.loc[key,"fplus_stale_delta_mean"]) < 1e-14
    oof = np.load(MOD / "oof_arrays.npz",allow_pickle=False)
    assert predictions.columns.tolist() == oof["targets"].tolist()
    j = oof["targets"].tolist().index("stale_hops")
    manifest = json.loads((MOD / 'final_artifact/manifest.json').read_text())
    assert complete['model'] == manifest['chosen_cv_candidate']
    assert complete['model'] in oof.files
    truth, estimate = oof['Y'][:,j], oof[complete['model']][:,j]
    r2 = float(1-np.sum((truth-estimate)**2)/np.sum((truth-truth.mean())**2))
    assert abs(r2-complete["selected_model_stale_oof_r2"]) < 1e-14
    after = {name:sha(FINAL/name) for name in complete["source_summary_sha256"]}
    assert before == after
    batching = json.loads((CHALLENGE / "query_batching_audit.json").read_text())
    assert batching['all_rows_exact_equal'] and batching['rows'] == len(chemistry)
    checks.to_csv(HERE / "deployment_transfer_independent_checks.csv",index=False)
    sources = [CHALLENGE / name for name in ["COMPLETE.json","cross_model_case_checks.csv",
        "all50_sensory_predictions.csv","source_and_candidate_chemistry.csv","query_batching_audit.json"]]
    sources += [MOD/"oof_arrays.npz",FINAL/"selection_cases.csv",FINAL/"selected_profiles.csv"]
    output = dict(completed=True,passed=True,timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        cases=len(joined),actual_csv_input_rows=len(chemistry),chemical_features=231,sensory_targets=50,
        unique_source_candidate_pairs=len(unique),nonincrease_unique_pairs=int(unique.nonincrease.sum()),
        increase_unique_pairs=int((~unique.nonincrease).sum()),
        unique_candidate_profiles=int(complete['unique_candidate_profiles']),
        identity_scope='Independent recount and exact input/output joins for this prepared run, not equality to historical outcome counts.', 
        maximum_chemical_identity_error=max_chemical_error,
        maximum_prediction_difference_error=float(checks.residual.abs().max()),
        stale_oof_r2_recomputed=r2,original_pool_unchanged=True,
        primary_nonincrease=groups.reset_index().loc[lambda d:np.isclose(d.rho,.1),["policy","nonincrease","cases"]].to_dict("records"),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},implementation_sha256=sha(__file__),
        scope="Only recomputes identities, saved CSV differences, event counts and OOF R2; no new model inference or experiment.")
    (HERE/"deployment_transfer_independent_audit.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    lines = ["# 最终CSV跨预测器挑战：独立结果复算", "",
        f"从本次CSV的 {len(chemistry)}×50 感官输出重算 {len(joined)} 个政策—源—ρ事件，逐项连接冻结候选和231维化学输入。{len(unique)} 个唯一源—候选比较中，{int(unique.nonincrease.sum())} 个非增、{int((~unique.nonincrease).sum())} 个增加；实际各政策结果列于独立检查表。原f★判定与候选身份逐项核验。", "",
        f"原始化学身份最大差{max_chemical_error:.3g}；候选减源的陈旧味差值最大复算残差{checks.residual.abs().max():.3g}。直接从250行OOF数组重算新部署模型的stale R²={r2:.9f}，与报告一致。输入50目标顺序、全部候选UID和原f★差值均一致，最终池哈希未变化。", "",
        "该检查复算保存结果，不重做神经网络推断；官方预处理逐行/整批等价由另一个保存审计记录。新预测器使用全部250款上下文，源啤酒在其中。模型间同向和分歧均属于内部代理模型敏感性，未提供真实酿造效果或独立人体感官证据。", ""]
    (HERE/"deployment_transfer_independent_review.md").write_text("\n".join(lines))
    print(json.dumps({k:output[k] for k in ["completed","passed","cases","unique_source_candidate_pairs","increase_unique_pairs","maximum_chemical_identity_error","maximum_prediction_difference_error"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
