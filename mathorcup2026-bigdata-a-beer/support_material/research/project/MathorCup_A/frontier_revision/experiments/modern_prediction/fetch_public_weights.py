"""Fetch the two frozen public model files, with exact SHA256 verification.

Uses Python's standard library; no account token or implicit latest revision.

"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
MODELS = {
    'tabicl': {
        'url': 'https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-regressor-v2-20260212.ckpt',
        'path': 'models/tabicl-regressor-v2-20260212.ckpt',
        'sha256': '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a',
        'bytes': 114324594,
    },
    'tabdpt': {
        'url': 'https://huggingface.co/Layer6/TabDPT/resolve/a5ca6e01c0fa09ec68c73e958e5199d1932abb3a/tabdpt1_3.safetensors',
        'path': 'tabdpt/models/tabdpt1_3.safetensors',
        'sha256': '97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb',
        'bytes': 252233296,
    },
}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main(args):
    for name in MODELS if args.model == 'all' else [args.model]:
        model = MODELS[name]
        target = HERE / model['path']
        if not target.exists():
            if args.verify_only:
                raise FileNotFoundError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + '.download')
            request = urllib.request.Request(model['url'], headers={'User-Agent': 'MathorCup-reproducibility/1.0'})
            with urllib.request.urlopen(request, timeout=60) as source, partial.open('wb') as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
            assert partial.stat().st_size == model['bytes'], 'Download size differs from frozen manifest.'
            assert digest(partial) == model['sha256'], 'Download SHA256 differs from frozen manifest.'
            partial.replace(target)
        assert target.stat().st_size == model['bytes'], f'Existing file size mismatch: {target}'
        assert digest(target) == model['sha256'], f'Existing file hash mismatch: {target}'
        print(json.dumps({'model': name, 'path': str(target), 'sha256': model['sha256'], 'verified': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['all', *MODELS], default='all')
    parser.add_argument('--verify-only', action='store_true')
    main(parser.parse_args())
