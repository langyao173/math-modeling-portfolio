#!/usr/bin/env python3
"""Create an isolated, empty-result research replay without editing provenance.

The historical training scripts record absolute source paths and platform
identity in immutable protocols. A fresh development-shaped workspace preserves
their source code, generates new protocols, and never rewrites delivered ones.
Only the original contest/frozen-baseline project is reused as scientific input.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import shutil
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def source_frontier(project):
    candidate = project / 'frontier_revision'
    if candidate.is_dir():
        return candidate
    candidate = project.parents[1]
    if (candidate / 'experiments/sparse_transfer/control_problem.py').is_file():
        return candidate
    raise FileNotFoundError('Cannot locate the delivered or development research revision.')


MODEL_INPUTS = {
    'models/tabicl-regressor-v2-20260212.ckpt',
    'models/TabICL_LICENSE.txt', 'models/tabicl_hf_metadata.json',
    'models/tabicl_download_manifest.json', 'models/checkpoint_configuration.json',
    'tabdpt/models/tabdpt1_3.safetensors', 'tabdpt/models/LICENSE',
    'tabdpt/models/checkpoint_header_metadata.json',
    'tabdpt/models/download_manifest.json', 'tabdpt/models/hf_metadata.json',
    'tabdpt/pretraining_audit.json', 'replay_project_inputs.json',
}
STATIC_NAMES = {
    'README.md', 'cold_reproduction_inventory.md', 'independent_transfer_audit.md',
    'pretraining_audit.md', 'batching_semantics.md',
    'perspective_method.md', 'perspective_sdp_method.md',
    'penalized_em_derivation.md',
}


def hidden_or_runtime(relative):
    return any(part.startswith('.') or part.startswith('venv') or part in
               {'__pycache__', 'node_modules', 'font_cache'} for part in relative.parts)


def experiment_source(relative):
    if hidden_or_runtime(relative):
        return False
    if relative.suffix == '.py' or relative.name in STATIC_NAMES:
        return True
    if relative.name.startswith('requirements') and relative.suffix == '.txt':
        return True
    if relative.name == 'environment_freeze.txt':
        return True  # Historical installation reference, never a current record.
    if relative.parts[0] == 'modern_prediction':
        return str(Path(*relative.parts[1:])) in MODEL_INPUTS
    return False


def prepare(project, destination):
    project, destination = Path(project).resolve(), Path(destination).resolve()
    frontier = source_frontier(project)
    if destination == project or project in destination.parents or destination in project.parents:
        raise ValueError('The replay workspace must be separate from the original project.')
    if destination.exists() and any(destination.iterdir()):
        marker = destination / 'REPLAY_SOURCE_MANIFEST.json'
        if marker.is_file():
            saved = json.loads(marker.read_text())
            if saved['original_project_identity'] != digest(project / 'data/A/1.csv'):
                raise ValueError('Existing replay belongs to different input data.')
            # Resume leaves fresh protocols and completed results in place.
            return destination / 'project/MathorCup_A', destination, False
        raise FileExistsError('Use an empty directory; no existing directory is cleaned or overwritten.')
    destination.mkdir(parents=True, exist_ok=True)
    records = []

    def copy(source, relative, category):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(dict(path=str(relative), category=category,
                            bytes=target.stat().st_size, sha256=digest(target)))

    baseline = Path('project/MathorCup_A')
    for source in sorted(project.rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(project)
        if hidden_or_runtime(relative) or relative.parts[0] == 'frontier_revision':
            continue
        if relative.parts[0] in {'reports', 'exemplar_reviews', 'delivery'}:
            continue
        if relative.name in {'frontier_delivery_manifest.json', 'frontier_staging_manifest.json'}:
            continue
        copy(source, baseline / relative, 'original_contest_and_frozen_baseline')
    for source in sorted((frontier / 'experiments').rglob('*')):
        if source.is_file() and experiment_source(source.relative_to(frontier / 'experiments')):
            copy(source, source.relative_to(frontier), 'revision_source_or_public_weight')
    for folder in ('audit', 'visuals', 'paper', 'literature'):
        for source in sorted((frontier / folder).rglob('*')):
            if not source.is_file():
                continue
            relative = source.relative_to(frontier / folder)
            if hidden_or_runtime(relative):
                continue
            keep = (folder == 'audit' and source.suffix == '.py')
            keep |= (folder == 'visuals' and len(relative.parts) == 1
                     and source.suffix in {'.py', '.md'})
            keep |= (folder == 'paper' and (
                (len(relative.parts) == 1 and source.suffix in {'.py', '.md'})
                or relative.parts[0] in {'history', 'tools'}))
            keep |= folder == 'literature'
            if keep:
                copy(source, source.relative_to(frontier), 'revision_static_source')
    for name in ('requirements_base_freeze.txt', 'runtime_versions.json'):
        if (frontier / name).is_file():
            copy(frontier / name, Path(name), 'historical_environment_reference')
    value = dict(format_version=1, original_project_identity=digest(project / 'data/A/1.csv'),
        source_project=str(project), source_revision=str(frontier),
        policy='New experiment directories contain source/static inputs and public weights only; no revision predictions, fitted models, specifications, audits or completion flags are copied.',
        baseline_policy='Original attachments, source programs and frozen baseline artifacts are retained as reference inputs. Reproducing these older layers uses reproduce_competition_revision.py separately.',
        files=records)
    (destination / 'REPLAY_SOURCE_MANIFEST.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
    return destination / baseline, destination, True


def interpreter_alias(directory, interpreter):
    """Compatibility entry for immutable runners that invoke venv/bin/python."""
    interpreter = shutil.which(str(interpreter)) or str(Path(interpreter).resolve())
    if not Path(interpreter).is_file():
        raise FileNotFoundError(interpreter)
    path = Path(directory) / 'bin/python'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Shell quoting is applied explicitly; the supplied path is never evaluated.
    path.write_text('#!/bin/sh\nexec ' + shlex.quote(interpreter) + ' "$@"\n')
    path.chmod(0o755)
    return dict(alias=str(path), interpreter=interpreter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    project, frontier, created = prepare(args.project, args.destination)
    print(json.dumps(dict(project=str(project), frontier=str(frontier), created=created), indent=2))
