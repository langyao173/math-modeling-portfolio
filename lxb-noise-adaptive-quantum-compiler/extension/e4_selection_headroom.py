"""Extension E4 — how much does candidate selection leave on the table, and does exact noisy
evaluation recover it?

The submitted compiler generates a pool of valid candidate compilations per circuit and selects one in two
steps: a reliability proxy (sum of log(1 - error) over gates and readouts, plus Pareto guards) ranks the pool,
then the proxy's choice and the top 8 are re-scored by a 512-shot noisy simulation with common random numbers
and the proxy's choice is replaced only if another candidate beats it by more than 0.006 (2.5-s budget).
Because the 512-shot estimate has a standard error of about 0.015-0.02 at the success levels involved, this
rule can both keep a worse candidate and pick a lucky one.

E4 evaluates every candidate's *exact* expected success under the evaluator's own noise model (the
density matrix of the noisy circuit — depolarising channels after every gate and bit-flip readout, exactly
as in the organisers' cirq_runner — summed over the bitstrings the benchmark accepts) and compares four
selection rules on the same pools:

  proxy only          the reliability proxy with Pareto guards (select_candidate)
  submitted           the shipped proxy + 512-shot common-random-number rule
  exact, top 8        exact expected success over the proxy's choice and its top 8 (same pool the submitted
                      rule looks at; same simulation cost as one 512-shot run, since the shot sampler draws
                      from this density matrix)
  exact, all          exact expected success over the whole pool (the best the pool allows)

Score of a rule = exact expected success of the candidate it selects (no evaluation shot noise).
Circuits: the 8 public Track A circuits, S1 (46 circuits from the official generator at unseen sizes) and
S2 (11 QASMBench circuits), built by extension/build_suites.py.  Circuits whose candidates act on more than
12 qubits, or whose exact evaluation exceeds 60 s for any candidate (dense 12-qubit bodies), are reported but
not scored exactly.  Finished circuits are cached in <suites_dir>/_e4_cache.jsonl so that a run can resume.
Run:  python extension/e4_selection_headroom.py <suites_dir>      (suites_dir from build_suites.py)
"""
from __future__ import annotations
import json, signal, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import cirq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from noise_adaptive_hackathon.benchmarks.io import load_benchmark                           # noqa: E402
from noise_adaptive_hackathon.formats.backend import CompetitionBackend                     # noqa: E402
from noise_adaptive_hackathon.simulator.cirq_runner import _inject_cirq_noise                # noqa: E402
from noise_adaptive_hackathon.submission import compiler as C                               # noqa: E402
from noise_adaptive_hackathon.submission.selection import select_candidate, rank_candidates_by_proxy   # noqa: E402
from noise_adaptive_hackathon.submission.simulation_selector import (                        # noqa: E402
    SELECTOR_TRACK_A_COMMON_RANDOM, SimulationSelectorConfig, select_candidate_with_simulation)

OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
MAX_EXACT_QUBITS = 12
CANDIDATE_TIMEOUT_S = 60     # a circuit whose exact evaluation exceeds this for one candidate is not scored


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()
CFG = SimulationSelectorConfig(mode=SELECTOR_TRACK_A_COMMON_RANDOM, track_a_top_k=8, track_a_lowshot=512,
                               track_a_timeout_sec=2.5, track_a_margin=0.006)


def make_compiler():
    return C.NoiseAdaptiveCompiler(
        enable_parity_resynthesis=True, enable_exact_small_router=True, enable_route_scar_normalizer=True,
        enable_measurement_tail_pruning=True, enable_relative_phase_toffoli_rewrite=True,
        enable_measurement_channel_physical_placement_v2=True, measurement_channel_physical_placement_v2_limit=2,
        enable_ghz_leaf_tree=True, enable_readout_copyout=True, enable_ghz_relabel=True,
        selector_mode=SELECTOR_TRACK_A_COMMON_RANDOM, selector_config=CFG)


def candidate_pool(comp, circuit, backend, seed=7):
    cands = comp.candidate_compilations(circuit, backend, seed)
    if comp.enable_measurement_tail_pruning:
        C._extend_candidates(cands, C._measurement_tail_pruned_candidates(circuit, backend, cands))
    if comp.enable_readout_copyout:
        C._extend_candidates(cands, C.readout_copyout_candidates(circuit, backend, cands))
    return cands


def exact_success(circuit: cirq.Circuit, backend: CompetitionBackend, rule) -> float | None:
    noisy = _inject_cirq_noise(circuit, backend)
    meas = [op for op in noisy.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if len(meas) != 1:
        return None
    mq = list(meas[0].qubits)
    body = cirq.Circuit(op for op in noisy.all_operations() if not isinstance(op.gate, cirq.MeasurementGate))
    others = sorted(set(body.all_qubits()) - set(mq))
    order = mq + others
    if len(order) > MAX_EXACT_QUBITS:
        return None
    rho = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(body, qubit_order=order).final_density_matrix
    p = np.real(np.diag(rho)).reshape([2] * len(order))
    p = p.sum(axis=tuple(range(len(mq), len(order)))) if others else p
    p = p.ravel(); p = np.clip(p, 0, None); p /= p.sum()
    nb = len(mq); acc = 0.0
    for idx in np.nonzero(p > 1e-14)[0]:
        bits = format(int(idx), f"0{nb}b")
        if rule.is_correct(bits):
            acc += p[idx]
    return float(acc)


def main():
    suites = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "extension" / "suites"
    roots = {"public": ROOT, **{p.name: p for p in sorted(suites.iterdir()) if p.is_dir()}}
    rows, cand_rows = [], []
    cache_path = suites / "_e4_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); cache[(r["row"]["suite"], r["row"]["benchmark"])] = r
    signal.signal(signal.SIGALRM, _alarm)
    for sname, root in roots.items():
        for bdir in sorted((root / "benchmarks" / "track_a" / "public").iterdir()):
            if (sname, bdir.name) in cache:
                rows.append(cache[(sname, bdir.name)]["row"]); cand_rows.extend(cache[(sname, bdir.name)]["cands"])
                continue
            b = load_benchmark(bdir)
            backend = CompetitionBackend.parse_raw((root / "backends" / b.metadata["backend_size_class"] /
                                                    f"{b.metadata['backend_snapshot_id']}.json").read_text(encoding="utf-8"))
            comp = make_compiler(); t0 = time.time()
            pool = candidate_pool(comp, b.circuit, backend)
            t_pool = time.time() - t0
            proxy = select_candidate(pool, backend, b.circuit)
            ranked = rank_candidates_by_proxy(pool, backend, b.circuit)
            t0 = time.time()
            submitted = C._protect_legacy_parity(select_candidate_with_simulation(pool, backend, b.circuit, seed=7, config=CFG),
                                                 pool, backend, b.circuit)
            t_sel = time.time() - t0
            ex, t_ex, timed_out = {}, [], False
            for c in pool:
                t1 = time.time()
                signal.setitimer(signal.ITIMER_REAL, CANDIDATE_TIMEOUT_S)
                try:
                    ex[id(c)] = exact_success(c.circuit, backend, b.expected_output_rule)
                except _Timeout:
                    timed_out = True
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                t_ex.append(time.time() - t1)
                if timed_out:
                    break
            if timed_out:
                ex = {id(c): None for c in pool}
            top8 = C_dedupe([proxy, *ranked[:8]])
            valid = [c for c in pool if ex[id(c)] is not None]
            scoreable = ex[id(proxy)] is not None and ex[id(submitted)] is not None and len(valid) == len(pool)
            row = dict(suite=sname, benchmark=b.name, family=b.metadata.get("family", ""), pool=len(pool),
                       compiled_qubits_max=max(len(c.circuit.all_qubits()) for c in pool),  # qubits touched by any compiled candidate, idle placeholders included t_pool=t_pool, t_submitted_select=t_sel,
                       t_exact_mean=float(np.mean(t_ex)), scored=bool(scoreable),
                       not_scored_reason="" if scoreable else ("exact evaluation over 60 s per candidate" if timed_out else "more than 12 qubits in the noisy circuit"))
            if scoreable:
                e8 = max((c for c in top8 if ex[id(c)] is not None), key=lambda c: ex[id(c)])
                ea = max(valid, key=lambda c: ex[id(c)])
                row.update(proxy=ex[id(proxy)], submitted=ex[id(submitted)], exact_top8=ex[id(e8)], exact_all=ex[id(ea)],
                           submitted_changed_proxy=submitted is not proxy, rank_of_best_in_proxy=ranked.index(ea) if ea in ranked else -1)
            rows.append(row)
            cands = [dict(suite=sname, benchmark=b.name, exact=ex[id(c)], log_rel_proxy=C._candidate_score(c.circuit, backend),
                          two_qubit=C.count_two_qubit_gates(c.circuit), source=getattr(c, "name", getattr(c, "label", ""))) for c in pool]
            cand_rows.extend(cands)
            with open(cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"row": {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in row.items()}, "cands": cands}, default=float) + "\n")
            print(sname, b.name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in
                                  ("pool", "proxy", "submitted", "exact_top8", "exact_all", "compiled_qubits_max")}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "e4_selection_headroom.csv", index=False)
    pd.DataFrame(cand_rows).to_csv(OUT / "e4_candidate_pool.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    s = df[df.scored]
    summ = {"circuits": int(len(df)), "scored_exactly": int(len(s)),
            "mean_by_suite": s.groupby("suite")[["proxy", "submitted", "exact_top8", "exact_all"]].mean().round(4).to_dict(orient="index"),
            "mean_all": s[["proxy", "submitted", "exact_top8", "exact_all"]].mean().round(4).to_dict(),
            "exact_top8_minus_submitted": {"mean": float((s.exact_top8 - s.submitted).mean()),
                                           "better": int((s.exact_top8 > s.submitted + 1e-9).sum()),
                                           "worse": int((s.exact_top8 < s.submitted - 1e-9).sum())},
            "submitted_minus_proxy": {"mean": float((s.submitted - s.proxy).mean()), "better": int((s.submitted > s.proxy + 1e-9).sum()),
                                      "worse": int((s.submitted < s.proxy - 1e-9).sum())},
            "headroom_all_minus_submitted": float((s.exact_all - s.submitted).mean()),
            **proxy_quality(pd.DataFrame(cand_rows), s),
            "mean_exact_eval_seconds": float(df.t_exact_mean.mean()), "mean_submitted_select_seconds": float(df.t_submitted_select.mean())}
    (OUT / "e4_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def proxy_quality(cands, scored):
    """How well the raw reliability proxy (sum of log(1 - error), the ranking score) orders each pool by exact
    success: per-circuit Spearman correlation, whether its top candidate is the exact best, and the exact success
    of that top candidate (the simplest possible selector, without guards or simulation)."""
    from scipy.stats import spearmanr
    keep = set(zip(scored.suite, scored.benchmark)); rho, top1_best, top1_val, spread, med_gap = [], [], [], [], []
    for (su, be), g in cands.groupby(["suite", "benchmark"]):
        if (su, be) not in keep:
            continue
        g = g.dropna(subset=["exact"]); top = g.loc[g.log_rel_proxy.idxmax(), "exact"]; best = g.exact.max()
        top1_val.append(top); top1_best.append(top >= best - 1e-12)
        spread.append(best - g.exact.min()); med_gap.append(best - g.exact.median())
        if g.exact.nunique() > 1:
            rho.append(spearmanr(g.log_rel_proxy, g.exact)[0])
    return {"proxy_rank_spearman_median": float(np.median(rho)), "proxy_rank_spearman_min": float(np.min(rho)),
            "proxy_top1_is_exact_best_share": float(np.mean(top1_best)), "mean_exact_proxy_top1": float(np.mean(top1_val)),
            "pool_spread_best_minus_worst_mean": float(np.mean(spread)), "pool_best_minus_median_mean": float(np.mean(med_gap))}


def C_dedupe(cs):
    seen, out = set(), []
    for c in cs:
        if id(c) not in seen:
            seen.add(id(c)); out.append(c)
    return out


if __name__ == "__main__":
    main()
