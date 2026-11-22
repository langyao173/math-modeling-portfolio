"""One beer-weighted energy/variogram convention for the main generator table."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"experiments/sparse_transfer"))
from control_problem import PROJECT

OUT = HERE.parent/"experiments/generation_moments"
SOURCE = PROJECT/"A_work/q3"


def main():
    scores = pd.read_csv(SOURCE/"extension_outer_scores_by_beer.csv", float_precision="round_trip")
    folds = pd.read_csv(SOURCE/"extension_outer_comparison.csv", float_precision="round_trip")
    process = json.loads((SOURCE/"extension_selection_process.json").read_text())
    labels = dict(smooth_pooled="平滑高斯", interval_gaussian="区间高斯",
        interval_kernel_h03="区间核 h=.3", interval_kernel_h06="区间核 h=.6",
        interval_kernel_h09="区间核 h=.9", nested_selection_process="内层选模流程")
    rows = []
    for model in labels:
        ss = scores[scores.selected] if model == "nested_selection_process" else scores[scores.model.eq(model)]
        ff = folds[folds.selected] if model == "nested_selection_process" else folds[folds.model.eq(model)]
        assert len(ss) == 250 and ss.beer.nunique() == 250 and ss.outer_fold.nunique() == 3
        assert len(ff) == 3 and ff.outer_fold.nunique() == 3
        rows.append(dict(model=model, model_label=labels[model],
            evaluation_unit="beer", unique_validation_beers=250, outer_folds=3,
            energy=float(ss.energy.mean()), variogram=float(ss.variogram.mean()),
            energy_aggregation="Mean over all 250 held-out beers",
            c2st_auc=float(ff.c2st_auc.mean()), nn_ratio=float(ff.nn_ratio.mean()),
            exact_copy_rate=float(ff.exact_copy_rate.mean()),
            distribution_diagnostic_aggregation="Mean over three outer-fold diagnostic experiments",
            object_type="Outer-fold model-selection procedure" if model == "nested_selection_process" else "Fixed model family/configuration"))
    table = pd.DataFrame(rows)
    workflow = table[table.model.eq("nested_selection_process")].iloc[0]
    assert abs(workflow.energy-process["selected_nested_energy"]) < 1e-14
    assert abs(workflow.c2st_auc-process["selected_nested_c2st_auc"]) < 1e-14
    reference = pd.read_csv(OUT/"summary.csv").set_index("model").loc["interval_reference"]
    assert abs(table.set_index("model").loc["interval_gaussian", "energy"]-reference.beer_weighted_energy) < 1e-14
    table.to_csv(OUT/"generation_main_table.csv", index=False)
    metadata = dict(date="2026-10-24", source_project="Ancestor-discovered data/A/1.csv project",
        source_sha256={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in
            ["extension_outer_scores_by_beer.csv", "extension_outer_comparison.csv", "extension_selection_process.json"]},
        energy_and_variogram="Equal weight for each of the 250 real validation beers; fixed candidates and workflow use identical convention.",
        distribution_diagnostics="AUC, nearest-neighbour ratio and copy rate remain means of the three outer diagnostic experiments, explicitly distinguished from per-beer scores.",
        selected_each_outer=process["selected_each_outer"],
        final_generator="interval_gaussian, kappa=2, nu=20",
        no_new_model_fits=True)
    (OUT/"generation_main_table_protocol.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(table[["model", "energy", "variogram", "c2st_auc", "nn_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
