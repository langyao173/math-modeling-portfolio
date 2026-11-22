#!/usr/bin/env python3
"""Fetch and verify the fixed public CJK fonts used by the figure programs.

Uses only the Python standard library. The two fonts are unmodified official
Noto CJK releases under the SIL Open Font License 1.1. Run this program before
run_submission.py --prepare-only so the prepared project receives the fonts.
Existing files must match the registered identities; they are not overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
FILES = {
    'NotoSansCJKsc-Regular.otf': {
        'version': 'Noto Sans CJK SC 2.004',
        'url': 'https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf',
        'bytes': 16437364,
        'sha256': '2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b',
    },
    'NotoSerifCJKsc-Regular.otf': {
        'version': 'Noto Serif CJK SC 2.003',
        'url': 'https://raw.githubusercontent.com/notofonts/noto-cjk/Serif2.003/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf',
        'bytes': 24543080,
        'sha256': '2a2eae2628df83556c54018c41e20fa532c1b862c5256ae8b3f23feb918d12ca',
    },
    'LICENSE.txt': {
        'version': 'SIL Open Font License 1.1',
        'url': 'https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/LICENSE',
        'bytes': 4301,
        'sha256': '6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2',
    },
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify(path, specification):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != specification['bytes']:
        raise ValueError('Font or license size differs from the fixed release: ' + str(path))
    if digest(path) != specification['sha256']:
        raise ValueError('Font or license SHA-256 differs from the fixed release: ' + str(path))


def prepare(output, verify_only):
    if not verify_only:
        output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, specification in FILES.items():
        destination = output / name
        downloaded = False
        if not destination.exists() and not verify_only:
            temporary = output / (name + f'.{os.getpid()}.download')
            request = urllib.request.Request(specification['url'], headers={
                'User-Agent': 'MathorCup-A-reproducibility/1.0'})
            try:
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open('xb') as stream:
                    while block := response.read(1048576):
                        stream.write(block)
                verify(temporary, specification)
                if destination.exists():
                    verify(destination, specification)
                    temporary.unlink()
                else:
                    temporary.replace(destination)
                downloaded = True
            finally:
                temporary.unlink(missing_ok=True)
        verify(destination, specification)
        records.append(dict(file=name, path=str(destination), downloaded=downloaded,
                            verified=True, **specification))
    return dict(verified=True, output=str(output), verify_only=verify_only,
                font_software_modified=False, files=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=HERE / 'project/MathorCup_A/assets/fonts',
                        help='Font directory copied by run_submission.py into a new project.')
    parser.add_argument('--verify-only', action='store_true',
                        help='Check existing files without network access or file changes.')
    args = parser.parse_args()
    record = prepare(args.output.expanduser().resolve(), args.verify_only)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
