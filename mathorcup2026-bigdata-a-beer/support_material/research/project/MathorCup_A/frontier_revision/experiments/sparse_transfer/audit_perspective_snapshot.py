"""Preserve and verify recorded perspective evidence across final aggregation.

Existing INTEGRITY_DONE files and their hashes are never changed. NPZ array
fingerprints and ZIP metadata distinguish numerical from container changes.
Any recorded-file byte change is reported as a failure, even if arrays match.
"""
from pathlib import Path
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
import time
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'post_aggregation_integrity'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def npz_fingerprint(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: dict(shape=list(archive[key].shape), dtype=str(archive[key].dtype),
                    contiguous_bytes_sha256=hashlib.sha256(np.ascontiguousarray(archive[key]).tobytes()).hexdigest())
                  for key in archive.files}
    with zipfile.ZipFile(path) as container:
        members = [dict(filename=member.filename, date_time=list(member.date_time),
                    compress_type=member.compress_type, CRC=member.CRC,
                    file_size=member.file_size, compress_size=member.compress_size,
                    create_system=member.create_system, external_attr=member.external_attr,
                    flag_bits=member.flag_bits, extra_hex=member.extra.hex(), comment_hex=member.comment.hex())
                   for member in container.infolist()]
        comment = container.comment.hex()
    return dict(arrays=arrays, zip_members=members, zip_comment_hex=comment)


def case_files(directory):
    return sorted(str(path.relative_to(directory)) for policy in ('strict', 'exceedance')
                  for path in (directory / policy).glob('beer_*/*') if path.is_file())


def snapshot(name):
    directory = HERE / name
    gate_path = directory / 'INTEGRITY_DONE.json'
    gate = json.loads(gate_path.read_text())
    assert gate['state'] == 'complete_independently_verified'
    assert gate['completed_cases'] == 22 and gate['all_full_witnesses_pass']
    destination = OUT / f'{name}_before.json'
    if destination.exists():
        previous = json.loads(destination.read_text())
        assert previous['integrity_gate_sha256'] == sha(gate_path), 'Gate version changed; preserve and explain versions before making a new snapshot.'
        for relative, expected in previous['recorded_file_sha256'].items():
            assert sha(directory / relative) == expected, f'Recorded evidence changed since snapshot: {name}/{relative}'
        return dict(directory=name, snapshot_already_exists=True,
                    recorded_files=len(previous['recorded_file_sha256']))
    fingerprints = {}
    for relative, expected in gate['result_file_sha256'].items():
        path = directory / relative
        assert sha(path) == expected, f'Pre-aggregation gate mismatch: {name}/{relative}'
        if path.suffix == '.npz':
            fingerprints[relative] = npz_fingerprint(path)
    value = dict(directory=name, integrity_gate_sha256=sha(gate_path),
        recorded_file_sha256=gate['result_file_sha256'], npz_fingerprints=fingerprints,
        case_files_at_snapshot=case_files(directory),
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        purpose='Snapshot before the final aggregation; original integrity gate remains unchanged.')
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    return dict(directory=name, snapshot_already_exists=False,
                recorded_files=len(gate['result_file_sha256']), npz_files=len(fingerprints),
                snapshot_sha256=sha(destination))


def verify(name):
    directory = HERE / name
    before_path = OUT / f'{name}_before.json'
    before = json.loads(before_path.read_text())
    gate_path = directory / 'INTEGRITY_DONE.json'
    gate = json.loads(gate_path.read_text())
    gate_unchanged = before['integrity_gate_sha256'] == sha(gate_path)
    assert gate['result_file_sha256'] == before['recorded_file_sha256'], 'Integrity file list changed; do not silently replace the baseline.'
    mismatches = []
    for relative, expected in before['recorded_file_sha256'].items():
        path = directory / relative
        if not path.exists():
            mismatches.append(dict(file=relative, missing=True, expected_sha256=expected))
            continue
        actual = sha(path)
        if actual == expected:
            continue
        item = dict(file=relative, missing=False, expected_sha256=expected, actual_sha256=actual)
        if path.suffix == '.npz':
            current = npz_fingerprint(path)
            original = before['npz_fingerprints'][relative]
            item['arrays_bitwise_identical'] = current['arrays'] == original['arrays']
            item['zip_metadata_identical'] = current['zip_members'] == original['zip_members'] and current['zip_comment_hex'] == original['zip_comment_hex']
            item['current_npz_fingerprint'] = current
            item['classification'] = ('Container bytes changed; named arrays are bitwise identical.'
                if item['arrays_bitwise_identical'] else 'At least one array, shape, or dtype changed; numerical re-audit required.')
        mismatches.append(item)
    added = sorted(set(case_files(directory)) - set(before['case_files_at_snapshot']))
    return dict(directory=name, complete=True, recorded_files=len(before['recorded_file_sha256']),
        integrity_gate_unchanged=gate_unchanged, recorded_file_hashes_unchanged=not mismatches,
        passed=gate_unchanged and not mismatches, mismatches=mismatches,
        new_unrecorded_case_files={relative:sha(directory / relative) for relative in added},
        snapshot_sha256=sha(before_path), current_integrity_gate_sha256=sha(gate_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('snapshot', 'verify'), required=True)
    parser.add_argument('--directories', nargs='+', choices=('perspective', 'perspective_sdp'),
                        default=['perspective', 'perspective_sdp'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage == 'snapshot':
        print(json.dumps([snapshot(name) for name in args.directories], ensure_ascii=False, indent=2))
        return
    final_path = HERE / 'final_summary/COMPLETE.json'
    final = json.loads(final_path.read_text())
    assert all(final.get(key) is True for key in ('complete', 'all_perspective', 'sdp_finished'))
    reports = [verify(name) for name in args.directories]
    report = dict(stage='after_final_aggregation', complete=True,
        passed=all(record['passed'] for record in reports), directories=reports,
        final_summary_sha256=sha(final_path), selection_cases_sha256=sha(HERE / 'final_summary/selection_cases.csv'),
        source_pool_sha256=final['source_pool_sha256'], implementation_sha256=sha(__file__),
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        interpretation='Every previously recorded file is required to retain its exact hash. Newly added unrecorded canonical files are recorded separately and do not invalidate the original gate. No gate or recorded hash was rewritten.')
    assert report['selection_cases_sha256'] == report['source_pool_sha256']
    path = OUT / 'verification.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise RuntimeError('Post-aggregation evidence changed; inspect verification.json and preserve versions before any independent re-audit.')


if __name__ == '__main__':
    main()
