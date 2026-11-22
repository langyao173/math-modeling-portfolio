"""Independent checks of the completed stability artifact, without refitting.

This file is deliberately run only after final candidate scoring. It recomputes
metrics from saved draw-level arrays and verifies selected cached-model outputs.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import json
import sys
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"sparse_transfer"))
from control_problem import FixedProblem, PROJECT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((HERE/"run_summary.json").read_text())
    assert summary["completed"] and summary["replicates"] == 200
    protocol = json.loads((HERE/"protocol.json").read_text())
    frozen = json.loads((HERE/"candidate_freeze.json").read_text())
    final = HERE.parent / "sparse_transfer/final_summary"
    completion = json.loads((final / "COMPLETE.json").read_text())
    assert all(completion.get(k) is True for k in ("complete", "all_perspective", "sdp_finished"))
    integrity_path = HERE.parent / "sparse_transfer/post_aggregation_integrity/verification.json"
    integrity = json.loads(integrity_path.read_text())
    assert integrity.get("complete") is True and integrity.get("passed") is True
    assert sha(integrity_path) == frozen["post_aggregation_integrity_sha256"]
    assert integrity["final_summary_sha256"] == sha(final / "COMPLETE.json")
    assert integrity["selection_cases_sha256"] == sha(final / "selection_cases.csv")
    assert integrity["source_pool_sha256"] == completion["source_pool_sha256"]
    for key, value in frozen["source_sha256"].items():
        if key.startswith("final_pool_"):
            assert sha(final / key[len("final_pool_"):]) == value
    with np.load(HERE/"bootstrap_draws.npz", allow_pickle=False) as saved:
        draws = {key:saved[key] for key in saved.files}
    index = np.load(HERE/"bootstrap_indices.npz", allow_pickle=False)
    assert int(draws["completed"]) == 200 and np.array_equal(draws["sample_indices"], index["indices"])
    meta = pd.read_csv(HERE/"profile_catalog.csv")
    profiles = pd.read_csv(HERE/"frozen_profiles.csv", float_precision="round_trip")
    stability = pd.read_csv(HERE/"conditional_model_stability.csv", float_precision="round_trip").set_index("profile_id")
    cases = pd.read_csv(HERE/"case_comparisons.csv", float_precision="round_trip")
    raw = pd.read_csv(PROJECT/"data/A/1.csv")
    assert sha(PROJECT/"data/A/1.csv") == protocol["source_sha256"]["chemistry"]
    assert sha(PROJECT/"data/A/4.csv") == protocol["source_sha256"]["sensory"]
    assert sha(HERE/"frozen_profiles.csv") == frozen["frozen_profile_sha256"]
    assert np.array_equal(meta.profile_id.to_numpy(str), draws["profile_id"])
    assert np.array_equal(profiles.profile_id, meta.profile_id)
    assert len(meta) == summary["n_profiles"]
    assert not cases.duplicated(["policy", "beer", "rho"]).any() and len(cases) == 88
    assert int(cases.candidate_exists.sum()) == summary["selected_cases"]
    chosen = pd.read_csv(final / "selection_cases.csv", float_precision="round_trip")
    aligned = chosen.merge(cases, on=["policy", "beer", "rho"], suffixes=("_parent", "_scored"), validate="one_to_one")
    assert len(aligned) == 88 and (aligned.candidate_exists_parent == aligned.candidate_exists_scored).all()
    present = aligned.candidate_exists_parent
    assert (aligned.loc[present, "candidate_uid_parent"] == aligned.loc[present, "candidate_uid_scored"]).all()
    assert set(cases.loc[cases.candidate_exists, "profile_id"]).issubset(set(meta.profile_id))
    style = raw.tasting_category_fine.to_numpy(str)
    labels, counts = np.unique(style, return_counts=True)
    expected_indices = []
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    for r in range(200):
        expected_indices.append(np.concatenate([rng.choice(np.flatnonzero(style == s), int(n), replace=True)
            for s, n in zip(labels, counts)]))
        actual_labels, actual_counts = np.unique(style[index["indices"][r]], return_counts=True)
        assert np.array_equal(actual_labels, labels) and np.array_equal(actual_counts, counts)
    assert np.array_equal(np.stack(expected_indices), index["indices"])
    targets = list(draws["targets"])
    assert len(targets) == 50 and len(draws["families"]) == 6
    st = targets.index("stale_hops")
    ti = list(draws["classes"]).index("Tripel")
    component = draws["component_delta"]
    weighted = np.einsum("f,rfpq->rpq", draws["weights"], component)
    weighted_error = float(np.max(abs(weighted-draws["ensemble_delta"])))
    assert weighted_error < 1e-12
    assert np.isfinite(component).all() and np.isfinite(draws["ensemble_delta"]).all()
    assert np.isfinite(draws["class_probability"]).all()
    assert np.max(abs(draws["class_probability"].sum(axis=2)-1)) < 1e-12
    assert np.array_equal(draws["predicted_class"], draws["class_probability"].argmax(axis=2))
    source_positions = np.flatnonzero(meta.kind.eq("source"))
    assert len(source_positions) == 11
    assert np.max(abs(component[:, :, source_positions])) == 0.
    assert np.max(abs(draws["ensemble_delta"][:, source_positions])) == 0.
    tripel = draws["predicted_class"] == ti
    stale = draws["ensemble_delta"][:, :, st] <= 1e-8
    joint = tripel & stale
    metric_errors = {}
    for name, values in [("Tripel_frequency", tripel), ("stale_nonincrease_frequency", stale), ("joint_frequency", joint)]:
        error = float(np.max(abs(stability.loc[meta.profile_id, name].to_numpy()-values.mean(axis=0))))
        metric_errors[name] = error
        assert error < 1e-14
    assert (stability.loc[meta.loc[meta.kind.eq("source"), "profile_id"], "stale_nonincrease_frequency"] == 1).all()
    source_map = {int(meta.iloc[j].beer):j for j in source_positions}
    raw_lookup = {int(beer):j for j, beer in enumerate(raw.beer)}
    multiplicity = np.stack([np.bincount(ix, minlength=250) for ix in index["indices"]])
    for j, row in meta.iterrows():
        oob = multiplicity[:, raw_lookup[int(row.beer)]] == 0
        assert int(oob.sum()) == int(stability.loc[row.profile_id, "source_oob_replicates"])
        if oob.any():
            assert abs(joint[oob, j].mean()-stability.loc[row.profile_id, "source_oob_joint_frequency"]) < 1e-14
    cross_policy_count = 0
    for uid, g in meta.loc[meta.kind.eq("new")].groupby("candidate_uid"):
        positions = g.index.to_numpy()
        if len(positions) < 2:
            continue
        cross_policy_count += 1
        assert np.array_equal(component[:, :, positions[0]], component[:, :, positions[1]])
        assert np.array_equal(draws["class_probability"][:, positions[0]], draws["class_probability"][:, positions[1]])
    for row in cases.itertuples():
        assert bool(row.old80_meets_required_rho) == (row.old80_target_distance_reduction >= row.rho-1e-8)
        if row.candidate_exists:
            assert row.target_distance_reduction >= row.rho-1e-8
            assert abs(row.joint_frequency-stability.loc[row.profile_id, "joint_frequency"]) < 1e-14
            assert abs(row.joint_frequency_minus_old80-(row.joint_frequency-row.old80_joint_frequency)) < 1e-14
            assert bool(row.old80_meets_required_rho) == np.isfinite(row.joint_frequency_minus_old80_both_meet_rho)
    records = pd.read_csv(HERE/"bootstrap_fit_records.csv")
    sensory = records[(records.family != "style_classifier") & (records.family != "replicate_total")]
    assert len(sensory) == 1200 and (sensory.targets == 50).all()
    assert len(records[records.family.eq("style_classifier")]) == 200
    assert (sensory.groupby("replicate").size() == 6).all()
    # Independent prediction replay from two prefitted caches, all candidates,
    # all six sensory families, and all 50 targets. No model is fitted here.
    problem = FixedProblem("strict")
    query = profiles[problem.bundle["features"]].to_numpy(float)
    z_query = problem.control.map_.transform(profiles[problem.features].to_numpy(float))
    source_indices = np.array([source_map[int(b)] for b in meta.beer])
    replay = []
    for r in [0, 199]:
        cache_file = HERE/"model_cache"/f"replicate_{r:03d}.joblib"
        if not cache_file.exists():
            replay.append(dict(replicate=r, cache_available=False))
            continue
        sidecar = json.loads(cache_file.with_suffix(".json").read_text())
        assert sidecar["cache_sha256"] == sha(cache_file)
        cache = joblib.load(cache_file)
        assert np.array_equal(cache["indices"], index["indices"][r])
        assert cache["replicate"] == r
        assert cache["targets"] == targets
        probabilities = cache["classifier"].predict_proba(z_query)
        probability_error = float(np.max(abs(probabilities-draws["class_probability"][r])))
        sensory_error = 0.
        for k, estimator in enumerate(cache["models"]):
            assert estimator.y_mean_.shape == (50,)
            if problem.bundle["specs"][k]["family"] in ["extra_trees", "random_forest"]:
                assert estimator.est_.n_outputs_ == 50 and len(estimator.est_.estimators_) == 240
            prediction = estimator.predict(query)
            difference = prediction-prediction[source_indices]
            sensory_error = max(sensory_error, float(np.max(abs(difference-component[r, k]))))
            assert np.max(abs(prediction[source_positions]-draws["component_source_prediction"][r, k])) < 1e-12
        assert probability_error < 1e-14 and sensory_error < 1e-12
        replay.append(dict(replicate=r, cache_available=True, probability_max_error=probability_error,
            all_sensory_delta_max_error=sensory_error))
        del cache
    # Recompute empirical support thresholds independently with direct norms.
    support = pd.read_csv(HERE/"support_diagnostics.csv", float_precision="round_trip").set_index("profile_id")
    support_protocol = json.loads((HERE/"support_protocol.json").read_text())
    pca = joblib.load(HERE/"support_pca10.joblib")
    target_mask = problem.df.tasting_category_fine.eq("Tripel").to_numpy()
    assert target_mask.sum() == 28
    support_errors = {}
    for space, training, candidate in [("pca10", pca.transform(problem.z), pca.transform(z_query)),
                                       ("rank225", problem.z, z_query)]:
        reference = training[target_mask]
        distances = np.linalg.norm(reference[:, None, :]-reference[None, :, :], axis=2)/np.sqrt(training.shape[1])
        np.fill_diagonal(distances, np.inf)
        loo = np.sort(distances, axis=1)[:, :3].mean(axis=1)
        threshold = float(np.quantile(loo, .95, method="higher"))
        assert abs(threshold-support_protocol["thresholds"][space]) < 1e-12
        candidate_distance = np.linalg.norm(candidate[:, None, :]-reference[None, :, :], axis=2)/np.sqrt(training.shape[1])
        score = np.sort(candidate_distance, axis=1)[:, :3].mean(axis=1)
        error = float(np.max(abs(score-support.loc[meta.profile_id, space+"_mean3nn_distance"].to_numpy())))
        assert error < 1e-12
        support_errors[space] = error
    answer = dict(completed=True, date="2026-10-24", replicates=200, profiles=len(meta),
        selection_cases=len(cases), true_styles=22, targets_per_family=50, sensory_fits=1200,
        exact_stratified_indices_reproduced=True, source_hashes_unchanged=True,
        post_aggregation_integrity_passed_and_unchanged=True,
        all_final_pool_hashes_unchanged=True, final_selected_candidate_uids_reproduced=True,
        weighted_delta_max_error=weighted_error, stability_metric_max_errors=metric_errors,
        all_source_self_differences_zero=True, same_uid_cross_policy_groups_identical=cross_policy_count,
        paired_old80_rho_eligibility_checked=True, cache_prediction_replay=replay,
        empirical_support_max_errors=support_errors,
        audit_implementation_sha256=sha(Path(__file__)),
        candidate_freeze_sha256=sha(HERE/"candidate_freeze.json"),
        final_draws_sha256=sha(HERE/"bootstrap_draws.npz"),
        scope="Completed-artifact identities and numerical replay only; no physical validation or new bootstrap refits.")
    (HERE/"independent_checks.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2))
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
