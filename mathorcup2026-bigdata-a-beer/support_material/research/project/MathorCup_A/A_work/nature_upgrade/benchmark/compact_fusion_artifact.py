# Team review and independent work must be documented truthfully.
"""Verify, or apply once, the prediction-equivalent fusion storage reduction.

The frozen reference was saved from the original embedded, uncompressed-output
artifact before this migration.  No fit method is called by this script.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
UP = HERE.parent
FUSION = UP / 'fusion'
WORK = UP.parent
for path in [FUSION, UP/'adaptive', HERE, WORK/'q1']:
    sys.path.insert(0, str(path))
from fusion_model import (ResearchFusion, file_sha256, load_fusion,
                          make_old_bundle_reference)


def base_predictions(model, X):
    bundle = model.bundle
    old = bundle['old_bundle']
    values = [sum(w*m.predict(X) for w, m in zip(old['weights'], old['models']))]
    values += [m.predict(X) for m in bundle['adaptive']]
    values += [m.predict(X) for m in bundle['gbr_models']]
    if model.catboost is not None:
        values.append(model.catboost.predict(X))
    return np.stack(values)


def compare(model, reference):
    results = {}
    for label, X in [('full', reference['X']), ('probe', reference['probe'])]:
        P = base_predictions(model, X)
        expected = reference[label+'_bases']
        assert P.shape == expected.shape and np.isfinite(P).all()
        np.testing.assert_allclose(P, expected, rtol=0, atol=1e-12)
        result = dict(rows=len(X), branches=P.shape[0], all_finite=True,
                      base_bit_exact=bool(np.array_equal(P, expected)),
                      base_max_abs_difference=float(np.max(np.abs(P-expected))),
                      per_branch_max_abs_difference={name: float(np.max(np.abs(a-b)))
                          for name, a, b in zip(model.bundle['base_names'], P, expected)})
        for rule in ['global', 'target_shrink']:
            actual = model.predict(X, rule=rule)
            baseline = reference[label+'_'+rule]
            np.testing.assert_allclose(actual, baseline, rtol=0, atol=1e-12)
            result[rule] = dict(bit_exact=bool(np.array_equal(actual, baseline)),
                               max_abs_difference=float(np.max(np.abs(actual-baseline))))
        results[label] = result
    return results


def relocated_subprocess_check(pending, reference_path):
    """Check relative dependencies from a clean, independently located tree."""
    with tempfile.TemporaryDirectory(prefix='mathorcup_compact_fusion_') as td:
        package = Path(td)
        new_work = package/'A_work'
        new_up = new_work/'nature_upgrade'
        new_fusion = new_up/'fusion'
        copies = {
            FUSION/'fusion_model.py': new_fusion/'fusion_model.py',
            UP/'adaptive/adaptive_models.py': new_up/'adaptive/adaptive_models.py',
            HERE/'gbr_model.py': new_up/'benchmark/gbr_model.py',
            UP/'catboost/catboost_model.py': new_up/'catboost/catboost_model.py',
            WORK/'q1/sensory_model.py': new_work/'q1/sensory_model.py',
            WORK/'q1/sensory_ensemble.joblib': new_work/'q1/sensory_ensemble.joblib',
            pending: new_fusion/'final_fusion_models.joblib',
            FUSION/'catboost_branch.cbm': new_fusion/'catboost_branch.cbm',
            FUSION/'catboost_branch.json': new_fusion/'catboost_branch.json',
            reference_path: package/'reference.npz',
        }
        for source, destination in copies.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        child = r'''
import json, sys
from pathlib import Path
import numpy as np
root = Path.cwd()
sys.path.insert(0, str(root/'A_work/nature_upgrade/fusion'))
from fusion_model import load_fusion
model = load_fusion()
reference = np.load(root/'reference.npz')
result = {}
for label, X in [('full', reference['X']), ('probe', reference['probe'])]:
    result[label] = {}
    for rule in ['global', 'target_shrink']:
        actual = model.predict(X, rule=rule)
        expected = reference[label+'_'+rule]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        result[label][rule] = {'shape':list(actual.shape),
            'max_abs_difference':float(np.max(np.abs(actual-expected))),
            'bit_exact':bool(np.array_equal(actual, expected))}
print(json.dumps(result))
'''
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        # This optional directory supplies the installed third-party dependency;
        # every project source, model, and relative baseline reference is copied.
        vendor = UP/'vendor'
        if vendor.is_dir():
            env['PYTHONPATH'] = str(vendor)
        result = subprocess.run([sys.executable, '-c', child], cwd=package, env=env,
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply the one-time migration after verifying the original artifact hash.')
    args = parser.parse_args()
    reference_path = FUSION/'inference_compaction_reference.npz'
    reference_meta = json.loads((FUSION/'inference_compaction_reference.json').read_text())
    assert file_sha256(reference_path) == reference_meta['reference_sha256']
    reference = np.load(reference_path)
    current_path = FUSION/'final_fusion_models.joblib'
    if not args.apply:
        print(json.dumps(compare(load_fusion(FUSION), reference), indent=2))
        return
    assert file_sha256(current_path) == reference_meta['source_model_sha256'], 'The one-time source artifact has changed; refusing to replace it.'
    started = time.time()
    for path in [FUSION/'fusion_model.py', FUSION/'run_fusion.py', UP/'adaptive/adaptive_models.py']:
        ast.parse(path.read_text(), filename=str(path))
    legacy = load_fusion(FUSION)
    assert legacy.bundle['features'] == reference_meta['features']
    assert legacy.bundle['targets'] == reference_meta['targets']
    assert legacy.bundle['base_names'] == reference_meta['base_names']
    legacy_check = compare(legacy, reference)
    old_path = WORK/'q1/sensory_ensemble.joblib'
    baseline = joblib.load(old_path)
    original = legacy.bundle['old_bundle']
    assert [m.spec for m in baseline['models']] == [m.spec for m in original['models']]
    np.testing.assert_array_equal(baseline['weights'], original['weights'])
    assert baseline['features'] == original['features'] and baseline['targets'] == original['targets']
    np.testing.assert_array_equal(baseline['reference_y'], original['reference_y'])
    old_selection = json.loads((UP/'adaptive/full_data_old_selection.json').read_text())
    assert [m.spec for m in baseline['models']] == [old_selection['specs'][f] for f in old_selection['families']]
    np.testing.assert_array_equal(baseline['weights'], np.asarray(old_selection['weights']))
    payload = dict(legacy.bundle)
    payload.pop('old_bundle')
    payload['old_bundle_reference'] = make_old_bundle_reference(FUSION, old_path, baseline)
    payload['inference_storage'] = 'selected_outputs_with_shared_baseline_v1'
    layouts = []
    for selector in payload['adaptive']:
        before = sum(m.n_outputs_ for m in selector.models_.values())
        selector.compact_for_inference()
        selector.compact_for_inference()  # Explicit idempotency check.
        after = sum(m.n_outputs_ for m in selector.models_.values())
        assert after == len(selector.indices) == 50
        np.testing.assert_array_equal(np.sort(np.concatenate(list(selector.output_indices_.values()))), np.arange(50))
        layouts.append(dict(family=selector.models_[int(selector.indices[0])].spec['family'],
                            candidate_count=len(selector.models_), saved_target_states_before=before,
                            saved_target_states_after=after))
    in_memory = ResearchFusion(dict(payload, old_bundle=baseline), legacy.catboost)
    compact_check = compare(in_memory, reference)
    pending = FUSION/'final_fusion_models.compact.pending.joblib'
    joblib.dump(payload, pending, compress=3)
    serialized = joblib.load(pending)
    assert 'old_bundle' not in serialized
    assert serialized['old_bundle_reference'] == payload['old_bundle_reference']
    relocated_check = relocated_subprocess_check(pending, reference_path)
    before_bytes = current_path.stat().st_size
    after_bytes = pending.stat().st_size
    assert after_bytes < before_bytes
    # Replacement happens only after the independent-directory load succeeds.
    os.replace(pending, current_path)
    reloaded_check = compare(load_fusion(FUSION), reference)
    manifest_path = FUSION/'final_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(inference_storage=payload['inference_storage'],
                    old_bundle_reference=payload['old_bundle_reference'])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    report = dict(status='passed', no_retraining=True, absolute_tolerance=1e-12,
                  bytes_before=before_bytes, bytes_after=after_bytes,
                  bytes_saved=before_bytes-after_bytes,
                  reduction_fraction=1-after_bytes/before_bytes,
                  reference_sha256=reference_meta['reference_sha256'],
                  original_model_sha256=reference_meta['source_model_sha256'],
                  compact_model_sha256=file_sha256(current_path),
                  baseline_reference=payload['old_bundle_reference'],
                  adaptive_layout=layouts, legacy_embedded_format=legacy_check,
                  compact_in_memory=compact_check, compact_reloaded=reloaded_check,
                  independent_directory_subprocess=relocated_check,
                  source_sha256={str(p.relative_to(UP)):file_sha256(p)
                      for p in [FUSION/'fusion_model.py', FUSION/'run_fusion.py', UP/'adaptive/adaptive_models.py']},
                  elapsed_seconds=time.time()-started,
                  scope='Prediction-preserving inference storage change; does not revise validation metrics.')
    (FUSION/'inference_compaction_validation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
