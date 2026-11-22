#!/usr/bin/env python3
"""Restore the native, SHA-256-verified source appendix of a submission Word.

Uses only the Python standard library; refuses an existing output directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
import xml.etree.ElementTree as ET

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_sources(document):
    with zipfile.ZipFile(document) as archive:
        tree = ET.fromstring(archive.read('word/document.xml'))
    result = {}; name = None; expected = None; lines = []
    def finish():
        if name is None:
            return
        if expected is None:
            raise ValueError('Missing source checksum: ' + name)
        joined = '\n'.join(lines).encode('utf-8')
        choices = [joined, joined + b'\n']
        matches = [b for b in choices if digest(b) == expected]
        if len(matches) != 1:
            raise ValueError('Source checksum mismatch: ' + name)
        result[name] = matches[0]
    for paragraph in tree.findall('.//w:body/w:p', NS):
        style = paragraph.find('w:pPr/w:pStyle', NS)
        style = style.get(W + 'val', '') if style is not None else ''
        text = ''.join(t.text or '' for t in paragraph.iter(W + 't'))
        if style == 'MCSourceFile':
            finish()
            name = text.strip(); path = PurePosixPath(name)
            if (not name or path.is_absolute() or '..' in path.parts
                    or '\\' in name or ':' in name or name in result):
                raise ValueError('Unsafe or repeated source path: ' + name)
            expected = None; lines = []
        elif style == 'MCSourceHash' and name is not None:
            match = re.fullmatch(r'SHA-256:\s*([0-9a-f]{64})', text.strip())
            if match is None or expected is not None:
                raise ValueError('Invalid or repeated source checksum: ' + name)
            expected = match.group(1)
        elif style == 'MCSourceCode' and name is not None:
            lines.append(text)
    finish()
    if not result:
        raise ValueError('No native source appendix found')
    if 'SOURCE_MANIFEST.json' in result:
        manifest = json.loads(result['SOURCE_MANIFEST.json'])
        records = manifest.get('files', manifest.get('sources', []))
        if isinstance(records, dict):
            records = [dict(path=k, **v) if isinstance(v, dict)
                       else dict(path=k, sha256=v) for k, v in records.items()]
        for record in records:
            key = record.get('path', record.get('relative_path'))
            expected_hash = record.get('sha256')
            if key is None or expected_hash is None:
                raise ValueError('Unsupported source manifest record')
            if key not in result or digest(result[key]) != expected_hash:
                raise ValueError('Source manifest mismatch: ' + str(key))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('document', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError('Choose a new output directory')
    sources = read_sources(args.document)
    args.output_directory.mkdir(parents=True)
    for name, data in sources.items():
        destination = args.output_directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(json.dumps({'verified_files': len(sources),
                      'output_directory': str(args.output_directory.resolve())},
                     ensure_ascii=False))

if __name__ == '__main__':
    main()
