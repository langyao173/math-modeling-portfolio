"""Post-selection conditional model stability and empirical support challenge.

The 200 style-stratified resamples are frozen before candidate selection.
All six original sensory families are refitted on all 50 targets, with original
specifications and fixed ensemble weights. No robustness result is returned to
the primary support optimizer. Probabilities below describe bootstrap model
stability, never physical or sensory intervention success probabilities.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits, threadpool_info
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sparse_transfer"))
from control_problem import FixedProblem, PROJECT, AW, file_sha
from sensory_model import SensoryEstimator, predict_bundle

REPLICATES = 200
SEED = 2026102407
RHOS = [.001, .1, .25, .4]
POLICIES = ["strict", "exceedance"]
ATOL = 1e-8
FINAL_POOL = HERE.parent / "sparse_transfer/final_summary"
POST_AGGREGATION_GATE = HERE.parent / "sparse_transfer/post_aggregation_integrity/verification.json"
MIN_FREE_BYTES = 8 * 1024**3


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def freeze_resampling():
    HERE.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PROJECT / "data/A/1.csv")
    frame_y = pd.read_csv(PROJECT / "data/A/4.csv").set_index("beer").loc[frame.beer].reset_index()
    bundle = joblib.load(AW / "q1/sensory_ensemble.joblib")
    styles = frame.tasting_category_fine.to_numpy(str)
    rng = np.random.default_rng(SEED)
    indices = np.stack([np.concatenate([rng.choice(np.flatnonzero(styles == style),
        int(np.sum(styles == style)), replace=True) for style in np.unique(styles)]) for _ in range(REPLICATES)])
    source_sha = {"chemistry":file_sha(PROJECT / "data/A/1.csv"),
        "sensory":file_sha(PROJECT / "data/A/4.csv"),
        "frozen_sensory_bundle":file_sha(AW / "q1/sensory_ensemble.joblib"),
        "frozen_control":file_sha(AW / "q3/transfer_and_augmentation_models.joblib"),
        "frozen_classifier":file_sha(AW / "competition_revision/style_feasibility/frozen_style_classifier.joblib")}
    path = HERE / "bootstrap_indices.npz"
    if path.exists():
        old = np.load(path, allow_pickle=False)
        assert np.array_equal(indices, old["indices"])
        assert np.array_equal(frame.beer.to_numpy(), old["beer"])
    else:
        np.savez_compressed(path, indices=indices, beer=frame.beer.to_numpy(), styles=styles,
                            seed=SEED, replicates=REPLICATES)
        long = pd.DataFrame(dict(replicate=np.repeat(np.arange(REPLICATES), len(frame)),
            draw_position=np.tile(np.arange(len(frame)), REPLICATES),
            training_row_index=indices.ravel(),
            beer=frame.beer.to_numpy()[indices.ravel()], style=styles[indices.ravel()]))
        long.to_csv(HERE / "bootstrap_indices.csv", index=False)
    assert set(bundle["targets"]) == set(frame_y.columns[3:])
    protocol = dict(date="2026-10-24", question="How stable are fixed transfer profiles under resampling and how close are they to observed Tripel profiles?",
        replicates=REPLICATES, bootstrap_seed=SEED, n=250, n_styles=22,
        source_sha256=source_sha, index_sha256=file_sha(path),
        candidate_selection=dict(policies=POLICIES, rhos=RHOS,
            pool="Completed final_summary pooling primary search, perspective candidates, and full-checked strict witnesses under the relaxed policy.",
            completion_gate="final_summary/COMPLETE.json must have complete=True, all_perspective=True, sdp_finished=True; post_aggregation_integrity/verification.json must have complete=True, passed=True and matching final-pool hashes before candidate freezing or scoring.",
            rule="Within each policy/source/rho, frozen full-model accepted candidates meeting rho; minimum recorded raw_fields_changed, then minimum quadratic_objective, then stable candidate_uid.",
            tie_break="Final aggregation uses stable candidate_uid for exact edit-count/objective ties; no stability score enters selection.",
            no_feasible_candidate="Explicit missing selection; never replaced by source or an infeasible row.",
            comparisons="Each original source and its existing joint k=80 profile; old profiles retain their original acceptance protocol."),
        style_refit=dict(estimator="Multinomial LogisticRegression", C=.1, class_weight="balanced",
            solver="lbfgs", max_iter=3000, tolerance=.0001,
            preprocessing="Original control map fitted on the 250 observed beers is fixed; no control covariance or map refit.",
            success="argmax class == Tripel; stale_hops(candidate)-stale_hops(source) <= 1e-8"),
        sensory_refit=dict(specifications=bundle["specs"], weights=np.asarray(bundle["weights"]).tolist(),
            families=bundle["families"], targets=bundle["targets"],
            target_count=len(bundle["targets"]), imputer="Same original median imputation; refitted within each bootstrap sample.",
            preprocessing="All original sensory preprocessors and target means/scales refit on bootstrap rows.",
            trees="All 50 targets jointly fitted for both 240-tree models; fixed original random seed 20261023.",
            tuning="No hyperparameter, family, weighting, or optimization tuning from robustness scores."),
        resampling="Sample beers with replacement within each observed style; preserve all 22 style counts and pair all profiles and model families on the same replicate.",
        prefit="Training-only model caches may be fitted while the final pool is pending; no candidate or source robustness predictions are made before the completion gate and candidate identity freeze.",
        inference="Conditional stability of this frozen modeling procedure under empirical style-stratified resampling; not physical intervention probability, not independent external validation.",
        oob_diagnostic="Additional conditional summary restricted to resamples excluding that source beer; it does not undo original candidate selection using the full dataset.",
        support=dict(projection="Original rank control map, then PCA(10), unwhitened, full SVD fit on all 250 observed rows.",
            reference="The 28 observed Tripel profiles.",
            score="Mean Euclidean distance to 3 nearest target-style observations divided by sqrt(dimension).",
            threshold="95th empirical percentile, method=higher, of 28 leave-one-neighbour-out reference scores in the same fixed projection.",
            supplementary="The same 3NN diagnostic in all 225 rank coordinates and the PCA reconstruction residual; not used to select candidates.",
            interpretation="Descriptive support flag in a learned projection; no coverage, typicality, or feasibility probability guarantee."),
        cpu_threads=1, save="Every resampling index; all six-family 50-target candidate-source prediction differences; source predictions; class probabilities and margins; complete selection/profile identity map.")
    if (HERE / "protocol.json").exists():
        prior = json.loads((HERE / "protocol.json").read_text())
        assert prior["source_sha256"] == source_sha, "Training evidence changed after protocol freeze"
        assert prior["bootstrap_seed"] == SEED and prior["replicates"] == REPLICATES
    write_json(HERE / "protocol.json", protocol)
    return frame, frame_y, bundle, indices, protocol


def training_identity(bundle, protocol):
    signature = dict(source_sha256=protocol["source_sha256"],
        index_sha256=protocol["index_sha256"], specs=bundle["specs"],
        weights=np.asarray(bundle["weights"]).tolist(), targets=bundle["targets"],
        features=bundle["features"], imputers=[m.imputer for m in bundle["models"]],
        sensory_implementation_sha256=file_sha(AW / "q1/sensory_model.py"),
        classifier=dict(C=.1, class_weight="balanced", solver="lbfgs", max_iter=3000, tol=.0001),
        tree_random_state=20261023, tree_count=240, targets_jointly_fitted=50)
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), signature


def fit_models_once(x, y, z, style, indices, bundle, replicate):
    """Fit only: this function never receives or predicts a candidate profile."""
    records = []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf = LogisticRegression(C=.1, class_weight="balanced", max_iter=3000, solver="lbfgs", tol=.0001)
        begin = time.perf_counter()
        clf.fit(z[indices], style[indices])
        records.append(dict(replicate=replicate, family="style_classifier", targets=1,
            unique_training_beers=len(np.unique(indices)), seconds=time.perf_counter()-begin,
            classifier_iterations=int(np.max(clf.n_iter_))))
        models = []
        for k, spec in enumerate(bundle["specs"]):
            begin = time.perf_counter()
            estimator = SensoryEstimator(spec, imputer=bundle["models"][k].imputer).fit(x[indices], y[indices])
            assert estimator.y_mean_.shape == (50,)
            if spec["family"] in ["extra_trees", "random_forest"]:
                assert estimator.est_.n_outputs_ == 50 and len(estimator.est_.estimators_) == 240
            models.append(estimator)
            records.append(dict(replicate=replicate, family=spec["family"], targets=50,
                unique_training_beers=len(np.unique(indices)), seconds=time.perf_counter()-begin))
    return clf, models, records, dict(seconds=time.perf_counter()-started,
        warnings=" | ".join(str(w.message) for w in caught))


def prefit_models(frame, frame_y, bundle, indices, protocol, maximum=REPLICATES):
    cache = HERE / "model_cache"
    cache.mkdir(exist_ok=True)
    identity, signature = training_identity(bundle, protocol)
    write_json(HERE / "prefit_protocol.json", dict(training_identity=identity, signature=signature,
        candidate_information_used=False, predictions_computed=False,
        cpu_threads=1, minimum_free_disk_bytes=MIN_FREE_BYTES,
        schedule="Training-only caching during pending final optimization; all candidate scoring waits for the all-perspective completion gate."))
    x = frame[bundle["features"]].to_numpy(float)
    y = frame_y[bundle["targets"]].to_numpy(float)
    style = frame.tasting_category_fine.to_numpy(str)
    pack = joblib.load(AW / "q3/transfer_and_augmentation_models.joblib")
    z = pack["model"].map_.transform(frame[pack["features"]].to_numpy(float))
    assert y.shape == (250, 50)
    started = time.perf_counter()
    stopped_for_disk = False
    for r in range(min(maximum, REPLICATES)):
        path, sidecar = cache / f"replicate_{r:03d}.joblib", cache / f"replicate_{r:03d}.json"
        if path.exists() and sidecar.exists():
            metadata = json.loads(sidecar.read_text())
            assert metadata["training_identity"] == identity
            assert metadata["cache_bytes"] == path.stat().st_size
            continue
        free = shutil.disk_usage(HERE).free
        if free < MIN_FREE_BYTES + 512*1024**2:
            stopped_for_disk = True
            print(json.dumps(dict(prefit_paused_for_reserved_disk=True, prefitted=r, free_bytes=free)), flush=True)
            break
        clf, models, records, fit = fit_models_once(x, y, z, style, indices[r], bundle, r)
        temp = path.with_suffix(".pending")
        joblib.dump(dict(training_identity=identity, replicate=r, indices=indices[r],
            classifier=clf, models=models, fit_records=records, fit_info=fit,
            targets=bundle["targets"], families=bundle["families"]), temp, compress=3)
        os.replace(temp, path)
        metadata = dict(replicate=r, training_identity=identity, cache_bytes=path.stat().st_size,
            cache_sha256=file_sha(path), fit_info=fit, fit_records=records,
            candidate_information_used=False, predictions_computed=False)
        write_json(sidecar, metadata)
        del clf, models
        if (r+1) % 5 == 0 or r+1 == min(maximum, REPLICATES):
            paths = list(cache.glob("*.joblib"))
            print(json.dumps(dict(prefitted=len(paths), total=REPLICATES,
                cache_bytes=sum(p.stat().st_size for p in paths), free_bytes=shutil.disk_usage(HERE).free,
                elapsed_seconds=time.perf_counter()-started, latest_fit_seconds=fit["seconds"],
                predictions_computed=False)), flush=True)
    records = [json.loads(p.read_text()) for p in sorted(cache.glob("replicate_*.json"))]
    flat = [dict(replicate=r["replicate"], cache_bytes=r["cache_bytes"], cache_sha256=r["cache_sha256"],
                 fit_seconds=r["fit_info"]["seconds"], warnings=r["fit_info"]["warnings"]) for r in records]
    pd.DataFrame(flat).to_csv(HERE / "prefit_manifest.csv", index=False)
    write_json(HERE / "prefit_summary.json", dict(prefitted=len(records), required=REPLICATES,
        completed=len(records) == REPLICATES, training_identity=identity,
        cache_bytes=sum(r["cache_bytes"] for r in records), reserved_disk_bytes=MIN_FREE_BYTES,
        free_bytes=shutil.disk_usage(HERE).free, stopped_for_disk=stopped_for_disk,
        predictions_computed=False, candidate_information_used=False,
        elapsed_this_invocation_seconds=time.perf_counter()-started))
    if len(records) == REPLICATES:
        pending = HERE / "PREFIT_COMPLETE.pending"
        write_json(pending, dict(completed=True, prefitted=REPLICATES,
            training_identity=identity, predictions_computed=False,
            candidate_information_used=False,
            cache_bytes=sum(r["cache_bytes"] for r in records),
            free_bytes=shutil.disk_usage(HERE).free))
        os.replace(pending, HERE / "PREFIT_COMPLETE.json")


def verified_pool_gate(wait=False):
    """Keep candidate freeze after the independent post-aggregation hash check."""
    started = time.perf_counter()
    announced = False
    while True:
        try:
            completion = json.loads((FINAL_POOL / "COMPLETE.json").read_text())
            integrity = json.loads(POST_AGGREGATION_GATE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            completion, integrity = {}, {}
        final_ready = all(completion.get(k) is True for k in ("complete", "all_perspective", "sdp_finished"))
        if integrity.get("complete") is True and integrity.get("passed") is False:
            raise RuntimeError("The post-aggregation integrity verification failed; scoring is blocked")
        if final_ready and integrity.get("complete") is True and integrity.get("passed") is True:
            assert integrity["final_summary_sha256"] == file_sha(FINAL_POOL / "COMPLETE.json")
            assert integrity["selection_cases_sha256"] == file_sha(FINAL_POOL / "selection_cases.csv")
            assert integrity["source_pool_sha256"] == completion["source_pool_sha256"]
            return integrity
        if not wait:
            raise RuntimeError("The completed final pool and successful post-aggregation integrity gate are required")
        if not announced:
            print(json.dumps(dict(waiting_for="successful post-aggregation integrity gate", candidate_scoring_started=False)), flush=True)
            announced = True
        if time.perf_counter() - started > 6 * 3600:
            raise TimeoutError("Post-aggregation integrity gate remained pending for six hours")
        time.sleep(10)


def collect_profiles(frame, problem):
    verified_pool_gate()
    completion_file = FINAL_POOL / "COMPLETE.json"
    if not completion_file.exists():
        raise FileNotFoundError("Final combined candidate pool has not been marked complete")
    completion = json.loads(completion_file.read_text())
    if not all(completion.get(k) is True for k in ("complete", "all_perspective", "sdp_finished")):
        raise RuntimeError("Only the completed all-perspective candidate pool may be scored")
    rows, meta, cases = [], [], []
    used = set()
    sources = sorted(problem.contexts)
    for beer in sources:
        src = frame[frame.beer == beer].iloc[0].to_dict()
        uid = f"source_b{beer}"
        rows.append(src);meta.append(dict(profile_id=uid, kind="source", beer=beer,
            policy="source", candidate_id=-1, raw_fields_changed=0, quadratic_objective=problem.contexts[beer]["F0"],
            target_distance_reduction=0.));used.add(uid)
    oldpath = AW / "competition_revision/style_feasibility"
    old = pd.read_csv(oldpath / "joint_selected_profiles.csv", float_precision="round_trip")
    diagnostics = pd.read_csv(oldpath / "joint_selected_diagnostics.csv", float_precision="round_trip")
    old = old[old.allowed_coordinates == 80]
    diagnostics = diagnostics[diagnostics.allowed_coordinates == 80].set_index("beer")
    assert set(old.beer.astype(int)) == set(sources) and old.beer.nunique() == len(sources)
    for beer in sources:
        row = old[old.beer == beer].iloc[0].to_dict()
        d = diagnostics.loc[beer]
        assert bool(d.accepted), "Existing k=80 comparator must be its accepted joint profile"
        uid = f"old80_b{beer}"
        rows.append(row);meta.append(dict(profile_id=uid, kind="old80", beer=beer,
            policy="old_strict", candidate_id=int(d.candidate_index),
            raw_fields_changed=int(d.raw_direct_fields_changed), quadratic_objective=float(d.quadratic_objective),
            target_distance_reduction=float(d.target_distance_reduction)));used.add(uid)
    hashes = {"old80_profiles":file_sha(oldpath / "joint_selected_profiles.csv"),
              "old80_diagnostics":file_sha(oldpath / "joint_selected_diagnostics.csv")}
    for name in ["COMPLETE.json", "selection_cases.csv", "selected_profiles.csv", "selected_full_checks.csv",
                 "strict_audited_pool_checks.csv", "exceedance_audited_pool_checks.csv"]:
        hashes["final_pool_"+name] = file_sha(FINAL_POOL / name)
    selected_cases = pd.read_csv(FINAL_POOL / "selection_cases.csv", float_precision="round_trip")
    profiles = pd.read_csv(FINAL_POOL / "selected_profiles.csv", float_precision="round_trip")
    final_checks = pd.read_csv(FINAL_POOL / "selected_full_checks.csv", float_precision="round_trip")
    assert not selected_cases.duplicated(["policy", "beer", "rho"]).any()
    assert len(selected_cases) == len(POLICIES)*len(sources)*len(RHOS)
    keys = ["evaluation_policy", "candidate_uid"]
    assert not profiles.duplicated(keys).any() and not final_checks.duplicated(keys).any()
    assert set(map(tuple, profiles[keys].to_numpy())) == set(map(tuple, final_checks[keys].to_numpy()))
    profiles = profiles.set_index(keys, drop=False)
    final_checks = final_checks.set_index(keys, drop=False)
    for policy in POLICIES:
        checks = pd.read_csv(FINAL_POOL / f"{policy}_audited_pool_checks.csv", float_precision="round_trip")
        assert checks.candidate_uid.is_unique
        accepted = checks.accepted if checks.accepted.dtype == bool else checks.accepted.astype(str).str.lower().eq("true")
        for beer in sources:
            for rho in RHOS:
                eligible = checks[accepted & checks.beer.eq(beer) & (checks.target_distance_reduction >= rho-ATOL)]
                eligible = eligible.sort_values(["raw_fields_changed", "quadratic_objective", "candidate_uid"], kind="stable")
                parent_case = selected_cases[selected_cases.policy.eq(policy) & selected_cases.beer.eq(beer) & np.isclose(selected_cases.rho, rho, atol=1e-14)]
                assert len(parent_case) == 1
                case = parent_case.iloc[0].to_dict()
                assert bool(case["candidate_exists"]) == bool(len(eligible))
                case.update(source_profile_id=f"source_b{beer}", old80_profile_id=f"old80_b{beer}")
                if len(eligible):
                    chosen = eligible.iloc[0]
                    cuid = str(chosen.candidate_uid)
                    assert str(case["candidate_uid"]) == cuid, "Final-pool lexicographic selection does not reproduce"
                    row = profiles.loc[(policy, cuid)].to_dict()
                    cid = int(row["candidate_id"])
                    uid = f"new_{policy}__{cuid}"
                    assert bool(final_checks.loc[(policy, cuid), "accepted"])
                    case.update(profile_id=uid, candidate_id=cid, candidate_uid=cuid,
                        raw_fields_changed=int(chosen.raw_fields_changed),
                        quadratic_objective=float(chosen.quadratic_objective),
                        target_distance_reduction=float(chosen.target_distance_reduction))
                    if uid not in used:
                        rows.append(row);meta.append(dict(profile_id=uid, kind="new", beer=beer, policy=policy,
                            candidate_id=cid, candidate_uid=cuid, search_policy=row["search_policy"], search_family=row["search_family"],
                            raw_fields_changed=int(chosen.raw_fields_changed),
                            quadratic_objective=float(chosen.quadratic_objective),
                            target_distance_reduction=float(chosen.target_distance_reduction)));used.add(uid)
                cases.append(case)
    rows, meta, cases = pd.DataFrame(rows), pd.DataFrame(meta), pd.DataFrame(cases)
    rows.insert(0, "profile_id", meta.profile_id.to_numpy())
    assert len(rows) == len(meta) and rows.profile_id.is_unique
    rows.to_csv(HERE / "frozen_profiles.csv", index=False)
    meta.to_csv(HERE / "profile_catalog.csv", index=False)
    cases.to_csv(HERE / "selection_cases.csv", index=False)
    write_json(HERE / "candidate_freeze.json", dict(source_sha256=hashes,
        post_aggregation_integrity_sha256=file_sha(POST_AGGREGATION_GATE),
        frozen_profile_sha256=file_sha(HERE / "frozen_profiles.csv"),
        selected_cases=int(cases.candidate_exists.sum()), total_cases=len(cases),
        unique_profiles=len(rows), unique_new_profiles=int(meta.kind.eq("new").sum()),
        unique_new_candidate_uids=int(meta.loc[meta.kind.eq("new"), "candidate_uid"].nunique()),
        code_sha256=file_sha(Path(__file__)),
        policy="Candidate identity is frozen before any bootstrap prediction; training-only prefits do not see the candidate pool; scores never feed optimization."))
    return rows, meta, cases


def support_diagnostic(problem, profiles, meta):
    z = problem.z
    model = PCA(n_components=10, whiten=False, svd_solver="full").fit(z)
    projected, candidate = model.transform(z), model.transform(problem.control.map_.transform(profiles[problem.features].to_numpy(float)))
    truth_ids = np.flatnonzero(problem.df.tasting_category_fine.eq("Tripel"))
    assert len(truth_ids) == 28
    ids = problem.df.beer.to_numpy(int)[truth_ids]
    scales = {"pca10":(projected, candidate), "rank225":(z, problem.control.map_.transform(profiles[problem.features].to_numpy(float)))}
    result = meta.copy()
    reference_rows, nearest_rows = [], []
    thresholds = {}
    for name, (train, query) in scales.items():
        target = train[truth_ids]
        dd = cdist(target, target)/np.sqrt(train.shape[1]);np.fill_diagonal(dd, np.inf)
        nearest = np.argsort(dd, axis=1)[:, :3]
        scores = np.take_along_axis(dd, nearest, axis=1).mean(axis=1)
        threshold = float(np.quantile(scores, .95, method="higher"))
        thresholds[name] = threshold
        cq = cdist(query, target)/np.sqrt(train.shape[1])
        cnear = np.argsort(cq, axis=1)[:, :3]
        cscore = np.take_along_axis(cq, cnear, axis=1).mean(axis=1)
        result[name+"_mean3nn_distance"] = cscore
        result[name+"_empirical_threshold"] = threshold
        result[name+"_distance_ratio"] = cscore/threshold
        result[name+"_within_empirical_threshold"] = cscore <= threshold
        for i, beer in enumerate(ids):
            reference_rows.append(dict(space=name, beer=int(beer), score=float(scores[i]),
                threshold=threshold, within_empirical_threshold=bool(scores[i] <= threshold),
                neighbour_beers="|".join(map(str, ids[nearest[i]]))))
        for i, profile_id in enumerate(meta.profile_id):
            for rank, j in enumerate(cnear[i], 1):
                nearest_rows.append(dict(profile_id=profile_id, space=name, neighbour_rank=rank,
                    observed_Tripel_beer=int(ids[j]), distance=float(cq[i, j])))
    train_resid = np.linalg.norm(z-model.inverse_transform(projected), axis=1)/np.sqrt(z.shape[1])
    query_z = scales["rank225"][1]
    query_resid = np.linalg.norm(query_z-model.inverse_transform(candidate), axis=1)/np.sqrt(z.shape[1])
    residual_threshold = float(np.quantile(train_resid[truth_ids], .95, method="higher"))
    result["pca_reconstruction_residual"] = query_resid
    result["pca_residual_Tripel_empirical_threshold"] = residual_threshold
    result["pca_residual_within_empirical_threshold"] = query_resid <= residual_threshold
    result.to_csv(HERE / "support_diagnostics.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(HERE / "support_reference_LOO.csv", index=False)
    pd.DataFrame(nearest_rows).to_csv(HERE / "nearest_observed_Tripels.csv", index=False)
    joblib.dump(model, HERE / "support_pca10.joblib")
    write_json(HERE / "support_protocol.json", dict(pca_components=10, whiten=False,
        explained_variance_ratio=model.explained_variance_ratio_.tolist(),
        total_explained_variance=float(model.explained_variance_ratio_.sum()),
        reference_n=28, neighbour_count=3, thresholds=thresholds,
        threshold_quantile=.95, quantile_method="higher",
        residual_threshold=residual_threshold,
        LOO_scope="Excludes self from the neighbour pool; the unsupervised rank/PCA representation stays fixed on all observed data.",
        no_probability_guarantee=True, diagnostic_only=True))
    return result


def fixed_predictor_checks(problem, profiles, meta):
    prediction = predict_bundle(problem.bundle, profiles)
    zz = problem.control.map_.transform(profiles[problem.features].to_numpy(float))
    probs = problem.clf.predict_proba(zz)
    ti = list(problem.clf.classes_).index("Tripel")
    lookup = {int(beer):i for i, beer in enumerate(meta.beer) if meta.kind.iloc[i] == "source"}
    source_indices = np.array([lookup[int(beer)] for beer in meta.beer])
    st = list(problem.bundle["targets"]).index("stale_hops")
    delta = prediction.to_numpy()-prediction.to_numpy()[source_indices]
    out = meta.copy()
    out["frozen_predicted_style"] = problem.clf.predict(zz)
    out["frozen_Tripel_probability"] = probs[:, ti]
    logits = problem.clf.decision_function(zz)
    others = logits.copy();others[:, ti] = -np.inf
    out["frozen_Tripel_logit_margin"] = logits[:, ti]-others.max(axis=1)
    out["frozen_new_style_epsilon_met"] = out.frozen_Tripel_logit_margin >= problem.style_epsilon-ATOL
    out["frozen_stale_delta"] = delta[:, st]
    out["frozen_joint_condition"] = out.frozen_predicted_style.eq("Tripel") & (delta[:, st] <= ATOL)
    # Old k=80 profiles used a different historical margin protocol; retain that
    # comparator even if it does not meet the new optimization's 1e-5 margin.
    for policy in POLICIES:
        where = np.flatnonzero(meta.policy.eq(policy))
        if len(where):
            p = problem if policy == "strict" else FixedProblem(policy)
            checked, _ = p.check_rows(profiles.iloc[where].to_dict("records"))
            assert checked.accepted.all(), f"Persisted {policy} selected profile failed the full frozen check"
    out.to_csv(HERE / "frozen_model_checks.csv", index=False)
    pd.DataFrame(delta, index=pd.Index(meta.profile_id, name="profile_id"), columns=problem.bundle["targets"]).to_csv(HERE / "frozen_sensory_delta.csv")
    return source_indices


def dump_checkpoint(path, completed, arrays, identity):
    temp = path.with_name(path.stem+"_pending.npz")
    np.savez_compressed(temp, completed=completed, identity=np.array(identity), **arrays)
    os.replace(temp, path)


def summarize(meta, cases, arrays, indices, frame, bundle):
    st = list(bundle["targets"]).index("stale_hops")
    labels = arrays["predicted_class"]
    classes = arrays["classes"]
    ti = int(np.flatnonzero(classes == "Tripel")[0])
    tripel = labels == ti
    stale_delta = arrays["ensemble_delta"][:, :, st]
    stale = stale_delta <= ATOL
    joint = tripel & stale
    beer_to_index = {int(b):i for i, b in enumerate(frame.beer)}
    counts = np.zeros((REPLICATES, len(frame)), dtype=int)
    for r, ix in enumerate(indices):
        counts[r] = np.bincount(ix, minlength=len(frame))
    source_count = counts[:, [beer_to_index[int(b)] for b in meta.beer]]
    oob = source_count == 0
    rows = []
    for j, item in meta.iterrows():
        row = item.to_dict()
        row.update(bootstrap_replicates=REPLICATES,
            Tripel_frequency=float(tripel[:, j].mean()), stale_nonincrease_frequency=float(stale[:, j].mean()),
            joint_frequency=float(joint[:, j].mean()),
            stale_delta_mean=float(stale_delta[:, j].mean()),
            stale_delta_025=float(np.quantile(stale_delta[:, j], .025)),
            stale_delta_median=float(np.median(stale_delta[:, j])),
            stale_delta_975=float(np.quantile(stale_delta[:, j], .975)),
            Tripel_probability_mean=float(arrays["class_probability"][:, j, ti].mean()),
            source_oob_replicates=int(oob[:, j].sum()))
        for label, value in [("Tripel", tripel), ("stale_nonincrease", stale), ("joint", joint)]:
            row["source_oob_"+label+"_frequency"] = float(value[oob[:, j], j].mean()) if oob[:, j].any() else np.nan
        rows.append(row)
    stability = pd.DataFrame(rows)
    stability.to_csv(HERE / "conditional_model_stability.csv", index=False)
    # Preserve the full sensory trade-off in a directly readable table. These
    # are conditional model predictions; no target is retrospectively selected.
    delta = arrays["ensemble_delta"]
    qlo, qhi = np.quantile(delta, [.025, .975], axis=0)
    sensory = pd.DataFrame(dict(profile_id=np.repeat(meta.profile_id.to_numpy(), len(bundle["targets"])),
        target=np.tile(bundle["targets"], len(meta)),
        bootstrap_delta_mean=delta.mean(axis=0).ravel(),
        bootstrap_delta_median=np.median(delta, axis=0).ravel(),
        bootstrap_delta_025=qlo.ravel(), bootstrap_delta_975=qhi.ravel(),
        nonincrease_frequency=(delta <= ATOL).mean(axis=0).ravel()))
    sensory.to_csv(HERE / "all_sensory_delta_summary.csv", index=False)
    component_rows = []
    for family_index, family in enumerate(bundle["families"]):
        change = arrays["component_delta"][:, family_index, :, st]
        lo, hi = np.quantile(change, [.025, .975], axis=0)
        component_rows.append(pd.DataFrame(dict(profile_id=meta.profile_id, family=family,
            ensemble_weight=float(bundle["weights"][family_index]),
            stale_delta_mean=change.mean(axis=0), stale_delta_025=lo, stale_delta_975=hi,
            stale_nonincrease_frequency=(change <= ATOL).mean(axis=0))))
    pd.concat(component_rows, ignore_index=True).to_csv(HERE / "family_stale_stability.csv", index=False)
    perdraw = []
    for r in range(REPLICATES):
        perdraw.append(pd.DataFrame(dict(replicate=r, profile_id=meta.profile_id,
            beer=meta.beer, predicted_style=classes[labels[r]],
            Tripel_probability=arrays["class_probability"][r, :, ti],
            Tripel_margin=arrays["Tripel_margin"][r], stale_delta=stale_delta[r],
            is_Tripel=tripel[r], stale_nonincrease=stale[r], joint=joint[r],
            source_bootstrap_multiplicity=source_count[r])))
    pd.concat(perdraw, ignore_index=True).to_csv(HERE / "bootstrap_prediction_differences.csv", index=False)
    columns = ["Tripel_frequency", "stale_nonincrease_frequency", "joint_frequency", "source_oob_joint_frequency",
               "stale_delta_mean", "stale_delta_025", "stale_delta_975"]
    case_results = cases.merge(stability[["profile_id"]+columns], how="left", on="profile_id", validate="many_to_one")
    for prefix in ["source", "old80"]:
        ref_columns = ["profile_id", "raw_fields_changed", "target_distance_reduction"]+columns
        ref = stability[ref_columns].rename(columns={c:prefix+"_"+c for c in ref_columns})
        case_results = case_results.merge(ref, how="left", on=prefix+"_profile_id", validate="many_to_one")
    case_results["old80_meets_required_rho"] = case_results.old80_target_distance_reduction >= case_results.rho-ATOL
    case_results["joint_frequency_minus_old80"] = case_results.joint_frequency-case_results.old80_joint_frequency
    case_results["joint_frequency_minus_old80_both_meet_rho"] = case_results.joint_frequency_minus_old80.where(
        case_results.candidate_exists & case_results.old80_meets_required_rho)
    case_results["old80_joint_frequency_on_selected"] = case_results.old80_joint_frequency.where(case_results.candidate_exists)
    case_results["source_joint_frequency_on_selected"] = case_results.source_joint_frequency.where(case_results.candidate_exists)
    case_results.to_csv(HERE / "case_comparisons.csv", index=False)
    grouping = case_results.groupby(["policy", "rho"], dropna=False)
    aggregate = grouping.agg(total_sources=("beer", "count"), selected_sources=("candidate_exists", "sum"),
        mean_edits=("raw_fields_changed", "mean"), mean_Tripel_frequency=("Tripel_frequency", "mean"),
        mean_stale_nonincrease_frequency=("stale_nonincrease_frequency", "mean"), mean_joint_frequency=("joint_frequency", "mean"),
        minimum_joint_frequency=("joint_frequency", "min"),
        mean_old80_joint_frequency=("old80_joint_frequency", "mean"),
        mean_source_joint_frequency=("source_joint_frequency", "mean"),
        mean_old80_joint_frequency_same_sources=("old80_joint_frequency_on_selected", "mean"),
        mean_source_joint_frequency_same_sources=("source_joint_frequency_on_selected", "mean"),
        mean_paired_joint_frequency_delta_old80=("joint_frequency_minus_old80", "mean"),
        old80_sources_meeting_required_rho=("old80_meets_required_rho", "sum"),
        mean_paired_joint_delta_both_meet_rho=("joint_frequency_minus_old80_both_meet_rho", "mean")).reset_index()
    aggregate["mean_scope"] = "New-model means condition on selected feasible candidates; same_sources comparator columns use that same subset. All-source comparator means and selected/total counts also retained."
    aggregate.to_csv(HERE / "aggregate_stability.csv", index=False)
    return stability, aggregate


def run(frame, frame_y, bundle, indices, protocol):
    started = time.perf_counter()
    verified_pool_gate(wait=True)
    problem = FixedProblem("strict")
    profiles, meta, cases = collect_profiles(frame, problem)
    support_diagnostic(problem, profiles, meta)
    source_indices = fixed_predictor_checks(problem, profiles, meta)
    x = frame[bundle["features"]].to_numpy(float)
    y = frame_y[bundle["targets"]].to_numpy(float)
    style = frame.tasting_category_fine.to_numpy(str)
    assert y.shape == (250, 50)
    query = profiles[bundle["features"]].to_numpy(float)
    z_query = problem.control.map_.transform(profiles[problem.features].to_numpy(float))
    classes = np.unique(style)
    ti = int(np.flatnonzero(classes == "Tripel")[0])
    n_profiles, n_targets, n_families = len(profiles), len(bundle["targets"]), len(bundle["specs"])
    assert n_families == 6 and n_targets == 50
    source_positions = np.flatnonzero(meta.kind.eq("source"))
    weights = np.asarray(bundle["weights"], float)
    fit_identity, _ = training_identity(bundle, protocol)
    identity = hashlib.sha256((file_sha(HERE / "candidate_freeze.json")+file_sha(HERE / "protocol.json")+file_sha(Path(__file__))).encode()).hexdigest()
    path = HERE / "bootstrap_checkpoint.npz"
    arrays = dict(component_delta=np.full((REPLICATES, n_families, n_profiles, n_targets), np.nan),
        component_source_prediction=np.full((REPLICATES, n_families, len(source_positions), n_targets), np.nan),
        ensemble_delta=np.full((REPLICATES, n_profiles, n_targets), np.nan),
        class_probability=np.full((REPLICATES, n_profiles, len(classes)), np.nan),
        Tripel_margin=np.full((REPLICATES, n_profiles), np.nan),
        predicted_class=np.full((REPLICATES, n_profiles), -1, dtype=np.int16),
        profile_id=meta.profile_id.to_numpy(str), source_profile_id=meta.profile_id.iloc[source_positions].to_numpy(str),
        targets=np.array(bundle["targets"]), families=np.array(bundle["families"]), classes=classes,
        weights=weights, sample_indices=indices)
    completed = 0
    if path.exists():
        stored = np.load(path, allow_pickle=False)
        assert str(stored["identity"]) == identity, "Checkpoint belongs to a different frozen task"
        completed = int(stored["completed"])
        arrays = {key:stored[key] for key in arrays}
    if (HERE / "bootstrap_draws.npz").exists():
        stored = np.load(HERE / "bootstrap_draws.npz", allow_pickle=False)
        assert str(stored["identity"]) == identity, "Completed run belongs to a different frozen task"
        assert int(stored["completed"]) == REPLICATES
        arrays = {key:stored[key] for key in arrays};completed = REPLICATES
    fit_records = pd.read_csv(HERE / "bootstrap_fit_records.csv").to_dict("records") if (HERE / "bootstrap_fit_records.csv").exists() else []
    fit_records = [row for row in fit_records if row["replicate"] < completed]
    for r in range(completed, REPLICATES):
        begin = time.perf_counter(); ix = indices[r]
        cache_path = HERE / "model_cache" / f"replicate_{r:03d}.joblib"
        from_cache = cache_path.exists()
        if from_cache:
            cache = joblib.load(cache_path)
            assert cache["training_identity"] == fit_identity
            assert np.array_equal(cache["indices"], ix) and cache["replicate"] == r
            assert cache["targets"] == bundle["targets"] and cache["families"] == bundle["families"]
            clf, models, records, fit_info = cache["classifier"], cache["models"], cache["fit_records"], cache["fit_info"]
            del cache
        else:
            clf, models, records, fit_info = fit_models_once(x, y, problem.z, style, ix, bundle, r)
        fit_records.extend(dict(record, training_only_prefit_cache=from_cache) for record in records)
        assert np.array_equal(clf.classes_, classes)
        pr = clf.predict_proba(z_query)
        decision = clf.decision_function(z_query)
        other = decision.copy();other[:, ti] = -np.inf
        arrays["class_probability"][r] = pr
        arrays["predicted_class"][r] = np.argmax(pr, axis=1)
        arrays["Tripel_margin"][r] = decision[:, ti]-np.max(other, axis=1)
        for k, estimator in enumerate(models):
            prediction = estimator.predict(query)
            assert prediction.shape == (n_profiles, 50)
            arrays["component_delta"][r, k] = prediction-prediction[source_indices]
            arrays["component_source_prediction"][r, k] = prediction[source_positions]
        arrays["ensemble_delta"][r] = np.einsum("f,fpq->pq", weights, arrays["component_delta"][r])
        assert np.isfinite(arrays["ensemble_delta"][r]).all()
        assert np.max(np.abs(arrays["ensemble_delta"][r, source_positions])) == 0.
        fit_records.append(dict(replicate=r, family="replicate_total", targets=50,
            unique_training_beers=len(np.unique(ix)), seconds=time.perf_counter()-begin,
            classifier_iterations=int(np.max(clf.n_iter_)),
            warnings=fit_info["warnings"], training_only_prefit_cache=from_cache,
            original_fit_seconds=fit_info["seconds"]))
        del models, clf
        if (r+1) % 5 == 0 or r+1 == REPLICATES:
            dump_checkpoint(path, r+1, arrays, identity)
            pd.DataFrame(fit_records).to_csv(HERE / "bootstrap_fit_records.csv", index=False)
            print(json.dumps(dict(completed=r+1, total=REPLICATES, elapsed_seconds=time.perf_counter()-started,
                latest_replicate_seconds=time.perf_counter()-begin, profiles=n_profiles)), flush=True)
    stability, aggregate = summarize(meta, cases, arrays, indices, frame, bundle)
    dump_checkpoint(HERE / "bootstrap_draws.npz", REPLICATES, arrays, identity)
    if path.exists():path.unlink()  # Identical completed arrays now have a final artifact name.
    codehash = file_sha(Path(__file__))
    write_json(HERE / "run_summary.json", dict(completed=True, replicates=REPLICATES,
        n_profiles=len(meta), n_new_profiles=int(meta.kind.eq("new").sum()),
        n_new_candidate_uids=int(meta.loc[meta.kind.eq("new"), "candidate_uid"].nunique()),
        selection_cases=len(cases), selected_cases=int(cases.candidate_exists.sum()),
        full_sensory_refits=REPLICATES*6, targets_per_fit=50,
        one_thread=True, threadpools=threadpool_info(),
        elapsed_seconds=time.perf_counter()-started, code_sha256=codehash,
        original_sources_unchanged=file_sha(PROJECT/"data/A/1.csv") == protocol["source_sha256"]["chemistry"],
        result_scope="Post-selection conditional model stability and descriptive data-support diagnostics, without feeding results to main optimization."))
    print(aggregate.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Freeze the bootstrap protocol and indices without waiting for primary optimization outputs")
    parser.add_argument("--prefit", action="store_true", help="Fit and compress training-only models without reading or scoring any candidate")
    parser.add_argument("--prefit-limit", type=int, default=REPLICATES, help="Maximum training-only replicate index to complete; default all 200")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        data = freeze_resampling()
        if args.prepare:
            print(json.dumps(dict(prepared=True, replicates=REPLICATES, index_sha256=file_sha(HERE / "bootstrap_indices.npz"))), flush=True)
        elif args.prefit:
            prefit_models(*data, maximum=args.prefit_limit)
        else:
            run(*data)


if __name__ == "__main__":
    main()
