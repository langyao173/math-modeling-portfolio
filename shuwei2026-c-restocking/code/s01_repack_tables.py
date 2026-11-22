# -*- coding: utf-8 -*-
"""Step 01. Copy every non-video member of the three raw archives into compact
table-only archives (work/interim/*_tables.zip).  Videos are not needed by the
numerical pipeline; payload bytes of the copied members are unchanged."""
import zipfile, os, time, hashlib, json
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
CROOT = os.path.dirname(WORK)
RAW = os.path.join(CROOT, 'Restocking_Problem_and_Data_EN', 'Raw_Data')
OUT = os.path.join(WORK, 'interim')
os.makedirs(OUT, exist_ok=True)
manifest = {}
for name in ['EgoDemo.zip', 'EgoPro.zip', 'EgoStandard.zip']:
    t0 = time.time()
    src = zipfile.ZipFile(os.path.join(RAW, name))
    dst_path = os.path.join(OUT, name.replace('.zip', '_tables.zip'))
    n = 0; tot = 0
    with zipfile.ZipFile(dst_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as dst:
        for info in src.infolist():
            if info.is_dir() or info.filename.lower().endswith('.mp4'):
                continue
            data = src.read(info.filename)
            dst.writestr(info.filename, data)
            manifest[info.filename] = {'bytes': len(data), 'crc32': info.CRC,
                                       'sha1': hashlib.sha1(data).hexdigest()}
            n += 1; tot += len(data)
    print('%s -> %s : %d members, %.1f MB raw, %.1f MB packed, %.1fs' % (
        name, os.path.basename(dst_path), n, tot / 1e6, os.path.getsize(dst_path) / 1e6, time.time() - t0), flush=True)
json.dump(manifest, open(os.path.join(OUT, 'table_member_manifest.json'), 'w'), indent=0)
print('done')
