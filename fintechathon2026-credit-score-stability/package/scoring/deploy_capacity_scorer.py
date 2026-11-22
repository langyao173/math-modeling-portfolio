"""Research-challenger scoring for FX50__g50, with one calibrated probability.

This explicit challenger API does not change or replace an incumbent service.

score_history rebuilds all 244+170 reference features from an account's complete
past accounting history. The lower-level initialize/advance API accepts those
features plus seven observed accounting fields. q is used for both review-list
ordering and score changes; quality_index is exactly 100*(1-q).

No banking service, LLM, or automatic credit decision is invoked.

Usage:
    scorer = CapacityChallengerScorer(package_root)
    state, result = scorer.initialize(observation)
    state, result = scorer.advance(state, next_observation)

Observation.timestamp is an integer MONTH ordinal, e.g. year*12+month-1.
Historical relative month indices are also accepted. A gap of more than one
month bypasses the historical gate. Persist the returned state per account and
reference period. A new customer, account, model, or reference period requires
an explicit initialize call. Calibration is applied AFTER temporal updating.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping
import argparse
import hashlib
import importlib.util
import json
import math
import sys

import joblib
import numpy as np
import pandas as pd


FINANCIAL = ('AMT_BALANCE', 'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_DRAWINGS_CURRENT',
             'AMT_INST_MIN_REGULARITY', 'AMT_RECEIVABLE_PRINCIPAL',
             'AMT_TOTAL_RECEIVABLE', 'AMT_RECIVABLE')
CANDIDATE = 'FX50__g50'
MODEL_VERSION = 'capacity-reference-20261002-fx50'


def number(x):
    return float('nan') if x is None else float(x)


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0., -x))


@dataclass(frozen=True)
class Observation:
    customer_id: str
    account_id: str
    model_version: str
    reference_period_id: str
    timestamp: int
    snapshot_id: str
    reference_limit: float
    reference_features: Mapping[str, float]
    financial: Mapping[str, float]
    current_dpd: float
    account_observations_to_date: int
    reference_supported: bool = True
    actual_credit_limit: float | None = None


@dataclass(frozen=True)
class ScoreState:
    customer_id: str
    account_id: str
    model_version: str
    model_fingerprint: str
    reference_period_id: str
    reference_limit: float
    timestamp: int
    snapshot_id: str
    financial: tuple
    current_dpd: float
    account_observations_to_date: int
    reference_supported: bool
    filtered_probability: float
    # Receipts support idempotent replays even after subsequent months arrive.
    # Each advance constructs a fresh dictionary and does not mutate its input.
    receipts: dict


class CapacityChallengerScorer:
    def __init__(self, package_root, model_version=MODEL_VERSION, calibration_path=None):
        self.root = Path(package_root)
        self.model_version = str(model_version)
        paths = [self.root/'models/baselines/final/reference_414_lgb.joblib',
                 self.root/'models/baselines/final/reference_244_xgb.joblib']
        self.models = [joblib.load(p) for p in paths]
        self.columns = list(self.models[0]['columns'])
        self.base_columns = list(self.models[1]['columns'])
        if len(self.columns) != 414 or len(self.base_columns) != 244:
            raise ValueError('Expected the frozen reference414 LightGBM and reference244 XGBoost.')
        if not set(self.base_columns).issubset(self.columns):
            raise ValueError('Base/reference feature contract mismatch.')
        if any(b.get('view') != 'reference' for b in self.models):
            raise ValueError('Both component models must consume the reference view.')
        cal_path = Path(calibration_path) if calibration_path is not None else self.root/'results_final/calibrations_final.json'
        self.calibration = json.loads(cal_path.read_text())[CANDIDATE]
        if not (math.isfinite(self.calibration['intercept'])
                and math.isfinite(self.calibration['coefficient'])
                and self.calibration['coefficient'] > 0):
            raise ValueError('Invalid frozen calibration.')
        digest = hashlib.sha256()
        feature_paths = [self.root/'feature_source/baseline/code/feature_engine.py',
                         self.root/'feature_source/code/augment_features.py',
                         self.root/'feature_source/baseline/data/feature_contract.json']
        for path in paths + feature_paths:
            digest.update(path.name.encode())
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(2**20), b''):
                    digest.update(chunk)
        digest.update(json.dumps(self.calibration, sort_keys=True).encode())
        self.model_fingerprint = digest.hexdigest()
        # Optional internal prediction cache, populated only by this model.
        self._cache = {}

    def _vector(self, obs):
        missing = set(self.columns) - set(obs.reference_features)
        if missing:
            raise ValueError(f'Missing reference feature columns: {sorted(missing)}')
        vector = np.array([number(obs.reference_features[c]) for c in self.columns], dtype='float32')
        if np.isinf(vector).any():
            raise ValueError('Infinite feature values are not valid model inputs.')
        return vector

    @staticmethod
    def _feature_key(vector):
        a = np.asarray(vector, dtype='float32').copy()
        a[np.isnan(a)] = np.nan  # Canonical NaN representation for the cache key.
        return hashlib.sha256(a.tobytes()).hexdigest()

    def raw_probabilities(self, frame, cache=False):
        """Run the frozen models on precomputed reference features in row order."""
        frame = pd.DataFrame(frame)[self.columns].astype('float32')
        if np.isinf(frame.to_numpy()).any():
            raise ValueError('Infinite reference feature values.')
        component_p = [b['model'].predict_proba(
            b['imputer'].transform(frame[b['columns']]).astype('float32'))[:, 1] for b in self.models]
        raw = .5 * (component_p[0] + component_p[1])
        if not np.isfinite(raw).all() or not ((raw >= 0) & (raw <= 1)).all():
            raise ValueError('Frozen components returned invalid probabilities.')
        if cache:
            for vector, p in zip(frame.to_numpy(), raw):
                self._cache[self._feature_key(vector)] = float(p)
        return raw

    def _raw(self, vector):
        key = self._feature_key(vector)
        if key in self._cache:
            return self._cache[key]
        frame = pd.DataFrame([vector], columns=self.columns)
        p = float(self.raw_probabilities(frame)[0])
        return p

    def _calibrated(self, probability):
        p = np.clip(float(probability), 1e-8, 1-1e-8)
        z = math.log(p) - math.log1p(-p)
        return float(_sigmoid(self.calibration['intercept'] + self.calibration['coefficient']*z))

    def _feature_modules(self):
        if hasattr(self, '_builders'):
            return self._builders
        engine_path = self.root/'feature_source/baseline/code/feature_engine.py'
        augment_path = self.root/'feature_source/code/augment_features.py'
        # augment_features imports feature_engine by this name. Multiple scorer
        # instances may share it only when they use the exact same source.
        if 'feature_engine' in sys.modules:
            module = sys.modules['feature_engine']
            original = Path(module.__file__)
            if hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(engine_path.read_bytes()).digest():
                raise ValueError('A different feature_engine version is already loaded.')
        else:
            spec = importlib.util.spec_from_file_location('feature_engine', engine_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules['feature_engine'] = module
            spec.loader.exec_module(module)
        name = '_capacity_augment_' + hashlib.sha256(augment_path.read_bytes()).hexdigest()[:12]
        if name in sys.modules:
            augmentation = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, augment_path)
            augmentation = importlib.util.module_from_spec(spec)
            sys.modules[name] = augmentation
            spec.loader.exec_module(augmentation)
        self._builders = module, augmentation
        return self._builders

    def observation_from_history(self, history, *, customer_id, account_id,
                                 reference_period_id, reference_limit, timestamp,
                                 snapshot_id, history_complete=True):
        """Recompute every reference descendant from one full account history.

        MONTHS_BALANCE and timestamp share an integer monthly coordinate.
        Rows after timestamp are excluded before constructing any feature.
        Missing historical months remain missing; they are never imputed as
        zero accounting activity. history_complete confirms that the supplied
        past records include all prior observations needed by ever/count fields.
        """
        if not history_complete:
            raise ValueError('Complete past account history is required for ever/count features.')
        if isinstance(timestamp, bool) or int(timestamp) != timestamp:
            raise ValueError('timestamp must be an integer month ordinal.')
        if not math.isfinite(reference_limit) or reference_limit <= 0:
            raise ValueError('reference_limit must be positive and finite.')
        engine, augmentation = self._feature_modules()
        raw = pd.DataFrame(history).copy()
        required = set(engine.NUM + ['SK_DPD', 'SK_DPD_DEF', 'MONTHS_BALANCE'])
        missing = required - set(raw)
        if missing:
            raise ValueError(f'Missing raw accounting columns: {sorted(missing)}')
        for column, expected in [('SK_ID_CURR', customer_id), ('SK_ID_PREV', account_id)]:
            if column in raw:
                values = raw[column].dropna().astype(str).unique()
                if len(values) != 1 or values[0] != str(expected):
                    raise ValueError(f'Raw history contains a different {column}.')
        months = pd.to_numeric(raw.MONTHS_BALANCE, errors='raise').to_numpy(float)
        if not np.isfinite(months).all() or not np.equal(months, np.floor(months)).all():
            raise ValueError('Raw history months must be finite integers.')
        raw['MONTHS_BALANCE'] = months.astype('int64')
        raw = raw.loc[raw.MONTHS_BALANCE.le(int(timestamp))].sort_values('MONTHS_BALANCE').reset_index(drop=True)
        if raw.MONTHS_BALANCE.duplicated().any():
            raise ValueError('One accounting row per account and month is required.')
        current = raw[raw.MONTHS_BALANCE.eq(int(timestamp))]
        if len(current) != 1:
            raise ValueError('The requested scoring month has no accounting record.')
        anchor = current.iloc[0]
        actual_limit = number(anchor.AMT_CREDIT_LIMIT_ACTUAL)
        financial = {c: number(anchor[c]) for c in FINANCIAL}
        # Feature builders use numeric account keys. The identity remains in
        # the observation/state; an internal single-account key also avoids
        # collisions when MONTHS_BALANCE is an absolute calendar month ordinal.
        raw['SK_ID_PREV'] = 1
        snapshots = pd.DataFrame([{'customer_id': str(customer_id), 'account_id': 1,
                                   'anchor_month': int(timestamp)}])
        base, _, columns = engine.features(raw, snapshots,
                                            reference_limits=np.array([float(reference_limit)]))
        transformed = raw.copy()
        transformed['AMT_CREDIT_LIMIT_ACTUAL'] = float(reference_limit)
        extra = augmentation.augment(transformed, snapshots)
        frame = pd.concat([base[columns].astype('float32'), extra], axis=1)
        if set(frame) != set(self.columns):
            raise ValueError('Rebuilt features differ from the frozen 414-column contract.')
        values = frame[self.columns].iloc[0].to_dict()
        count = int(values['account_observations_to_date'])
        return Observation(str(customer_id), str(account_id), self.model_version,
                           str(reference_period_id), int(timestamp), str(snapshot_id),
                           float(reference_limit), values, financial, number(anchor.SK_DPD),
                           count, count >= 6, actual_limit)

    def score_history(self, history, *, state=None, **snapshot):
        observation = self.observation_from_history(history, **snapshot)
        return self.initialize(observation) if state is None else self.advance(state, observation)

    def save_state(self, state, path):
        """Atomically persist account state as strict JSON, including receipts."""
        if state.model_fingerprint != self.model_fingerprint:
            raise ValueError('Cannot persist state from a different model fingerprint.')
        from dataclasses import asdict
        def finite(value):
            if isinstance(value, dict): return {k: finite(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)): return [finite(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value): return None
            return value
        _atomic_json(path, {'schema_version': 1, 'state': finite(asdict(state))})

    def load_state(self, path):
        payload = json.loads(Path(path).read_text())
        if payload.get('schema_version') != 1:
            raise ValueError('Unknown scoring-state schema version.')
        values = payload['state']
        values['financial'] = tuple(number(v) for v in values['financial'])
        values['current_dpd'] = number(values['current_dpd'])
        state = ScoreState(**values)
        if state.model_fingerprint != self.model_fingerprint or state.model_version != self.model_version:
            raise ValueError('Stored state uses a different model; explicit initialization is required.')
        if not math.isfinite(state.filtered_probability) or not 0 <= state.filtered_probability <= 1:
            raise ValueError('Stored pre-calibration probability is invalid.')
        return state

    def _validate(self, obs):
        for name in ('customer_id', 'account_id', 'model_version', 'reference_period_id', 'snapshot_id'):
            if getattr(obs, name) is None or not str(getattr(obs, name)).strip():
                raise ValueError(f'{name} must be present.')
        if str(obs.model_version) != self.model_version:
            raise ValueError('Model version changed; initialize with the intended model.')
        if isinstance(obs.timestamp, bool) or int(obs.timestamp) != obs.timestamp:
            raise ValueError('timestamp must be an integer month ordinal.')
        if not math.isfinite(obs.reference_limit) or obs.reference_limit <= 0:
            raise ValueError('A positive finite reference limit is required for initialization.')
        if int(obs.account_observations_to_date) != obs.account_observations_to_date or obs.account_observations_to_date < 0:
            raise ValueError('History observation count must be a nonnegative integer.')
        if set(FINANCIAL) - set(obs.financial):
            raise ValueError('All seven accounting keys are required; use None for missing values.')
        vector = self._vector(obs)
        features = dict(zip(self.columns, vector))
        if not np.isclose(features['account_observations_to_date'], obs.account_observations_to_date):
            raise ValueError('History count and reference features disagree.')
        # Current actual limits are never an input. Verify the precomputed
        # reference representation, including its historical limit levels.
        levels = ('anchor__', 'lag1__', 'w3_max__', 'w3_mean__', 'w12_max__', 'w12_mean__')
        for prefix in levels:
            value = features[prefix+'AMT_CREDIT_LIMIT_ACTUAL']
            if np.isfinite(value) and not np.isclose(value, obs.reference_limit, rtol=2e-6, atol=1e-3):
                raise ValueError('Precomputed features contain a non-reference credit limit.')
        for prefix in ('delta1__', 'anchor_minus_w3__', 'w3_minus_w12__'):
            value = features[prefix+'AMT_CREDIT_LIMIT_ACTUAL']
            if np.isfinite(value) and abs(value) > 1e-3:
                raise ValueError('Precomputed limit differences must reflect a fixed reference.')
        for lag in (2, 3, 4, 5):
            value = features[f'extra_lag{lag}_AMT_CREDIT_LIMIT_ACTUAL']
            if np.isfinite(value) and not np.isclose(value, obs.reference_limit, rtol=2e-6, atol=1e-3):
                raise ValueError('Augmented history contains a non-reference credit limit.')
        for window in (3, 6, 12):
            for name in (f'extra_limit_slope{window}', f'extra_limit_increases{window}',
                         f'extra_limit_decreases{window}'):
                value = features[name]
                if np.isfinite(value) and abs(value) > 1e-6:
                    raise ValueError('Augmented limit changes must reflect a fixed reference.')
        # Keep the observable gate and model input synchronized. Differences
        # allowed here are only float32 storage rounding, not stale evidence.
        values = tuple(number(obs.financial[c]) for c in FINANCIAL)
        for c, value in list(zip(FINANCIAL, values)) + [('SK_DPD', number(obs.current_dpd))]:
            anchor = features['anchor__'+c]
            both_missing = np.isnan(anchor) and np.isnan(value)
            if not both_missing and not (np.isfinite(anchor) and np.isfinite(value)
                                          and np.isclose(anchor, value, rtol=2e-6, atol=1e-3)):
                raise ValueError(f'Accounting evidence and reference features disagree: {c}.')
        return vector, values

    def _fingerprint(self, obs, vector, values):
        payload = {'customer': str(obs.customer_id), 'account': str(obs.account_id),
                   'version': obs.model_version, 'period': str(obs.reference_period_id),
                   'timestamp': int(obs.timestamp), 'snapshot': str(obs.snapshot_id),
                   'reference': float(obs.reference_limit),
                   'financial': [v if np.isfinite(v) else None for v in values],
                   'dpd': number(obs.current_dpd) if np.isfinite(number(obs.current_dpd)) else None,
                   'history': int(obs.account_observations_to_date),
                   'support': bool(obs.reference_supported),
                   'actual_limit': number(obs.actual_credit_limit) if np.isfinite(number(obs.actual_credit_limit)) else None,
                   'features_sha256': self._feature_key(vector)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _result(self, obs, raw, filtered, applied, reason):
        p = self._calibrated(filtered)
        return {'candidate': CANDIDATE, 'lifecycle_status': 'research_challenger',
                'customer_id': str(obs.customer_id), 'account_id': str(obs.account_id),
                'model_version': self.model_version, 'model_fingerprint': self.model_fingerprint,
                'reference_period_id': str(obs.reference_period_id),
                'timestamp': int(obs.timestamp), 'snapshot_id': str(obs.snapshot_id),
                'reference_limit': float(obs.reference_limit),
                'reference_supported': bool(obs.reference_supported and obs.account_observations_to_date >= 6),
                'raw_model_probability': float(raw),
                'filtered_probability_before_calibration': float(filtered),
                'q': p, 'risk_probability': p, 'reference_probability': p,
                'quality_index': 100*(1-p),
                'observed_credit_limit': number(obs.actual_credit_limit) if np.isfinite(number(obs.actual_credit_limit)) else None,
                'gate_applied': bool(applied), 'gate_reason': reason, 'idempotent_replay': False}

    def initialize(self, obs: Observation):
        vector, values = self._validate(obs)
        raw = self._raw(vector)
        output = self._result(obs, raw, raw, False, 'initialized')
        receipt = {str(obs.snapshot_id): {'fingerprint': self._fingerprint(obs, vector, values),
                                         'output': dict(output)}}
        state = ScoreState(str(obs.customer_id), str(obs.account_id), self.model_version,
                           self.model_fingerprint, str(obs.reference_period_id), float(obs.reference_limit),
                           int(obs.timestamp), str(obs.snapshot_id), values, number(obs.current_dpd),
                           int(obs.account_observations_to_date), bool(obs.reference_supported), raw, receipt)
        return state, output

    def advance(self, state: ScoreState, obs: Observation):
        if not math.isfinite(state.filtered_probability) or not 0 <= state.filtered_probability <= 1:
            raise ValueError('Stored pre-calibration probability is invalid; restore a valid state.')
        if len(state.financial) != len(FINANCIAL):
            raise ValueError('Stored accounting state does not match the seven-field contract.')
        expected = (state.customer_id, state.account_id, state.model_version, state.reference_period_id)
        actual = (str(obs.customer_id), str(obs.account_id), str(obs.model_version), str(obs.reference_period_id))
        if actual != expected or state.model_fingerprint != self.model_fingerprint:
            raise ValueError('Customer/account/model/reference period changed; explicit initialize is required.')
        if float(obs.reference_limit) != state.reference_limit:
            raise ValueError('Reference limit changed; explicit initialize of a new reference period is required.')
        vector, values = self._validate(obs)
        fingerprint = self._fingerprint(obs, vector, values)
        receipt = state.receipts.get(str(obs.snapshot_id))
        if receipt is not None:
            if receipt['fingerprint'] != fingerprint:
                raise ValueError('Same snapshot ID has conflicting evidence.')
            return state, dict(receipt['output'], idempotent_replay=True)
        if int(obs.timestamp) <= state.timestamp:
            raise ValueError('New evidence must strictly advance the monthly timestamp.')
        raw = self._raw(vector)
        reason = 'seven_fields_within_5pct'
        if int(obs.timestamp) != state.timestamp + 1:
            reason = 'nonadjacent_month'
        elif not (state.reference_supported and obs.reference_supported and
                  state.account_observations_to_date >= 6 and obs.account_observations_to_date >= 6):
            reason = 'insufficient_reference_history'
        elif not (np.isfinite(state.current_dpd) and np.isfinite(number(obs.current_dpd))):
            reason = 'missing_dpd'
        elif state.current_dpd != 0 or number(obs.current_dpd) != 0:
            reason = 'delinquency_observed'
        elif not (np.isfinite(state.financial).all() and np.isfinite(values).all()):
            reason = 'missing_accounting_evidence'
        else:
            delta = abs(np.asarray(values)-np.asarray(state.financial))
            normalized = np.r_[delta[:4]/max(state.reference_limit, 1.),
                                delta[4:]/state.reference_limit]
            if np.max(normalized) > .05:
                reason = 'accounting_change_exceeds_5pct'
        applied = reason == 'seven_fields_within_5pct'
        filtered = .5*state.filtered_probability + .5*raw if applied else raw
        output = self._result(obs, raw, filtered, applied, reason)
        receipts = dict(state.receipts)
        receipts[str(obs.snapshot_id)] = {'fingerprint': fingerprint, 'output': dict(output)}
        new_state = replace(state, timestamp=int(obs.timestamp), snapshot_id=str(obs.snapshot_id),
                            financial=values, current_dpd=number(obs.current_dpd),
                            account_observations_to_date=int(obs.account_observations_to_date),
                            reference_supported=bool(obs.reference_supported),
                            filtered_probability=float(filtered), receipts=receipts)
        return new_state, output


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def verify(package_root):
    """Replay stored feature rows; read frozen predictions, never outcomes."""
    root = Path(package_root)
    scorer = CapacityChallengerScorer(root)
    directory = root/'data/event_evaluation'
    meta = pd.read_parquet(directory/'meta.parquet')
    features = pd.concat([pd.read_parquet(directory/'reference.parquet'),
                          pd.read_parquet(directory/'reference_extra.parquet')], axis=1)
    extra = pd.read_parquet(directory/'actual.parquet', columns=['anchor__'+c for c in FINANCIAL[4:]])
    target = pd.read_parquet(root/'results_final/evaluation_predictions_frozen.parquet',
                            columns=['sample_id', CANDIDATE])
    assert np.array_equal(meta.sample_id, target.sample_id)
    raw = scorer.raw_probabilities(features, cache=True)
    observed = np.empty(len(meta)); gated = np.zeros(len(meta), bool)
    states, observations, outputs = {}, {}, {}
    for phase in (-1, 0, 1):
        for ix in np.flatnonzero(meta.phase.to_numpy() == phase):
            row = meta.iloc[ix]
            financial = {c: number(row[c]) for c in FINANCIAL[:4]}
            financial.update({c: number(extra.iloc[ix]['anchor__'+c]) for c in FINANCIAL[4:]})
            obs = Observation(str(row.customer_id), str(row.account_id), scorer.model_version,
                              str(row.event_row_id), int(row.anchor_month), str(row.sample_id),
                              float(row.reference_limit), features.iloc[ix].to_dict(), financial,
                              number(row.current_dpd), int(row.account_observations_to_date),
                              bool(row.reference_supported))
            key = str(row.event_row_id)
            if key in states:
                state, output = scorer.advance(states[key], obs)
            else:
                state, output = scorer.initialize(obs)
            states[key] = state
            observations[ix] = obs
            outputs[ix] = output
            observed[ix] = output['reference_probability']
            gated[ix] = output['gate_applied']
    error = abs(observed-target[CANDIDATE].to_numpy())
    assert np.max(error) < 1e-12, float(np.max(error))
    # Independently compare the observable seven-field gate to the frozen batch
    # gate; this source reads only feature/edge inputs, not outcomes.
    import metrics_previous
    original_root = metrics_previous.ROOT
    metrics_previous.ROOT = root
    try:
        edges = metrics_previous.complete_edges('event_evaluation')
    finally:
        metrics_previous.ROOT = original_root
    expected_gate = np.zeros(len(meta), dtype=bool)
    expected_gate[edges.loc[edges.seven_05, 'row_after'].to_numpy(int)] = True
    assert np.array_equal(gated, expected_gate)
    parity = {'candidate': CANDIDATE, 'model_fingerprint': scorer.model_fingerprint,
              'rows': len(meta), 'customers': int(meta.customer_id.nunique()),
              'phases': {str(k): int(v) for k, v in meta.phase.value_counts().items()},
              'max_probability_abs_error': float(np.max(error)),
              'mean_probability_abs_error': float(np.mean(error)),
              'gate_mismatches': int(np.sum(gated != expected_gate)),
              'gate_applied_rows': int(gated.sum()), 'evaluation_outcomes_read': False,
              'inference_input': '414 reconstructed or precomputed reference features plus seven accounting fields',
              'lifecycle_status': 'research_challenger'}
    _atomic_json(root/'results_final/streaming_challenger_parity.json', parity)

    # Real integration risks: use an observed gated pair and alter only the
    # explicitly tested payload/state contract. These are routing tests, not
    # predictive-performance experiments.
    edge = edges[edges.seven_05].iloc[0]
    before, after = int(edge.row_before), int(edge.row_after)
    old, new = observations[before], observations[after]
    initial, _ = scorer.initialize(old)
    current, result = scorer.advance(initial, new)
    tests = []
    def immediate(name, observation, expected_reason):
        _, out = scorer.advance(initial, observation)
        drift = abs(out['filtered_probability_before_calibration']-out['raw_model_probability'])
        assert not out['gate_applied'] and out['gate_reason'] == expected_reason and drift == 0
        tests.append({'case': name, 'passed': True, 'gate_reason': out['gate_reason'],
                      'raw_acceptance_abs_error': drift})
    def rejected(name, fn):
        try:
            fn()
        except ValueError as error:
            tests.append({'case': name, 'passed': True, 'message': str(error)})
        else:
            raise AssertionError(name+' should have been rejected')
    changed_features = dict(new.reference_features, anchor__SK_DPD=1.)
    immediate('new_delinquency', replace(new, current_dpd=1., reference_features=changed_features),
              'delinquency_observed')
    changed_financial = dict(new.financial)
    changed_financial['AMT_BALANCE'] = old.financial['AMT_BALANCE'] + .06*old.reference_limit
    changed_features = dict(new.reference_features, anchor__AMT_BALANCE=changed_financial['AMT_BALANCE'])
    immediate('balance_exceeds_threshold', replace(new, financial=changed_financial,
                                                  reference_features=changed_features),
              'accounting_change_exceeds_5pct')
    changed_financial = dict(new.financial, AMT_TOTAL_RECEIVABLE=None)
    changed_features = dict(new.reference_features, anchor__AMT_TOTAL_RECEIVABLE=np.nan)
    immediate('missing_receivable', replace(new, financial=changed_financial,
                                           reference_features=changed_features), 'missing_accounting_evidence')
    changed_features = dict(new.reference_features, account_observations_to_date=5.)
    immediate('insufficient_history', replace(new, account_observations_to_date=5,
                                            reference_features=changed_features),
              'insufficient_reference_history')
    immediate('reference_support_not_confirmed', replace(new, reference_supported=False),
              'insufficient_reference_history')
    immediate('missing_month', replace(new, timestamp=initial.timestamp+2), 'nonadjacent_month')
    repeated, same = scorer.advance(current, new)
    assert repeated is current and same['reference_probability'] == result['reference_probability']
    tests.append({'case': 'duplicate_evidence', 'passed': True,
                  'probability_abs_error': abs(same['reference_probability']-result['reference_probability']),
                  'state_identity_preserved': repeated is current})
    replayed, prior = scorer.advance(current, old)
    assert replayed is current and prior['idempotent_replay']
    tests.append({'case': 'earlier_evidence_replay', 'passed': True,
                  'state_timestamp_unchanged': replayed.timestamp == current.timestamp})
    rejected('out_of_order_new_snapshot', lambda: scorer.advance(current, replace(old, snapshot_id='new-stale-id')))
    rejected('conflicting_snapshot_payload', lambda: scorer.advance(current, replace(new, timestamp=new.timestamp+1)))
    rejected('new_reference_period', lambda: scorer.advance(current, replace(new, reference_period_id='new-period')))
    rejected('reference_limit_change', lambda: scorer.advance(current, replace(new, reference_limit=new.reference_limit*1.1)))
    rejected('cross_customer', lambda: scorer.advance(current, replace(new, customer_id='different-customer')))
    rejected('cross_account', lambda: scorer.advance(current, replace(new, account_id='different-account')))
    rejected('model_version_change', lambda: scorer.advance(current, replace(new, model_version='different-model')))
    reinitialized, restarted = scorer.initialize(replace(new, reference_period_id='new-period'))
    assert reinitialized.reference_period_id == 'new-period' and not restarted['gate_applied']
    tests.append({'case': 'explicit_reference_reinitialization', 'passed': True,
                  'new_period': reinitialized.reference_period_id})
    # Persist/reload state and compare its next transition, including receipts.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='capacity-state-') as temp:
        saved = Path(temp)/'account-state.json'
        scorer.save_state(initial, saved)
        restored = scorer.load_state(saved)
        restored_state, restored_result = scorer.advance(restored, new)
        assert restored_result == result and restored_state.filtered_probability == current.filtered_probability
        tests.append({'case': 'state_json_roundtrip_then_advance', 'passed': True})
        assert restored_result['q'] == restored_result['risk_probability'] == restored_result['reference_probability']
        assert restored_result['quality_index'] == 100*(1-restored_result['q'])
        tests.append({'case': 'single_probability_for_ranking_and_score', 'passed': True})
    _atomic_json(root/'results_final/streaming_challenger_checks.json',
                 {'checks': tests, 'n_checks': len(tests), 'all_passed': True,
                  'test_kind': 'state and routing checks; no outcome claims',
                  'external_services_called': False})
    print(json.dumps(parity, indent=2))
    print(json.dumps({'routing_checks': len(tests), 'all_passed': True}))
    return parity, tests


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify:
        parser.error('Use the Python initialize/advance API, or --verify for frozen replay checks.')
    verify(args.root)
