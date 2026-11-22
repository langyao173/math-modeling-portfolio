"""Portable inference for the Nature-source research extension.

"""
import sys
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_old_bundle_reference(directory, path, bundle):
    """Describe the already packaged baseline, without duplicating its models."""
    return dict(relative_path=os.path.relpath(Path(path).resolve(), Path(directory).resolve()),
                sha256=file_sha256(path), specs=[m.spec for m in bundle['models']],
                weights=np.asarray(bundle['weights']).tolist())


def resolve_old_bundle(bundle, directory):
    """Load a shared baseline or accept the original embedded bundle format."""
    if 'old_bundle' in bundle:
        return bundle['old_bundle']
    reference = bundle.get('old_bundle_reference')
    if reference is None:
        raise ValueError('Fusion bundle has neither an embedded nor a referenced baseline.')
    path = (Path(directory) / reference['relative_path']).resolve()
    if file_sha256(path) != reference['sha256']:
        raise ValueError('The referenced baseline file does not match its recorded SHA-256.')
    old = joblib.load(path)
    if old['features'] != bundle['features'] or old['targets'] != bundle['targets']:
        raise ValueError('Referenced baseline feature or target order differs from the fusion.')
    if [m.spec for m in old['models']] != reference['specs']:
        raise ValueError('Referenced baseline model specifications differ from the manifest.')
    if not np.array_equal(old['weights'], np.asarray(reference['weights'])):
        raise ValueError('Referenced baseline weights differ from the manifest.')
    if not np.array_equal(old['reference_y'], bundle['reference_y']):
        raise ValueError('Referenced baseline training targets differ from the fusion.')
    return old


class ResearchFusion:
    def __init__(self, bundle, catboost=None):
        self.bundle = bundle
        self.catboost = catboost

    def predict(self, frame, rule='target_shrink'):
        b = self.bundle
        X = frame[b['features']].to_numpy(float) if isinstance(frame, pd.DataFrame) else np.asarray(frame, float)
        if X.ndim != 2 or X.shape[1] != len(b['features']):
            raise ValueError('Supply exactly 231 chemical columns in the manifest feature order.')
        old = b['old_bundle']
        preds = [sum(w*m.predict(X) for w, m in zip(old['weights'], old['models']))]
        preds += [m.predict(X) for m in b['adaptive']]
        preds += [m.predict(X) for m in b['gbr_models']]
        if self.catboost is not None:
            preds.append(self.catboost.predict(X))
        P = np.stack(preds)
        if rule == 'global':
            return np.einsum('m,mij->ij', b['global_weights'], P)
        if rule != 'target_shrink':
            raise ValueError('rule must be global or target_shrink')
        return np.einsum('mj,mij->ij', b['target_weights'], P)


def load_fusion(directory=None):
    folder = Path(directory).resolve() if directory is not None else Path(__file__).resolve().parent
    upgrade = folder.parent
    for p in [upgrade/'adaptive', upgrade/'benchmark', upgrade/'catboost', upgrade.parent/'q1']:
        sys.path.insert(0, str(p))
    b = joblib.load(folder/'final_fusion_models.joblib')
    b['old_bundle'] = resolve_old_bundle(b, folder)
    cb = None
    if 'catboost_global' in b['base_names']:
        from catboost_model import CatboostMultiTarget
        cb = CatboostMultiTarget.load_native(folder, 'catboost_branch')
    return ResearchFusion(b, cb)
