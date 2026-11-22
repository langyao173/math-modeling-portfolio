# -*- coding: utf-8 -*-
"""List the contents of the three raw archives without extracting them."""
import zipfile, os, sys, collections, json
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
CROOT = os.path.dirname(WORK)
RAW = os.path.join(CROOT, 'Restocking_Problem_and_Data_EN', 'Raw_Data')
out = {}
for name in ['EgoDemo.zip', 'EgoPro.zip', 'EgoStandard.zip']:
    p = os.path.join(RAW, name)
    z = zipfile.ZipFile(p)
    infos = z.infolist()
    ext = collections.Counter()
    size = collections.Counter()
    enc = 0
    for i in infos:
        e = os.path.splitext(i.filename)[1].lower() if not i.is_dir() else '<dir>'
        ext[e] += 1
        size[e] += i.file_size
        enc += (i.flag_bits & 1)
    print('=====', name, 'entries', len(infos), 'encrypted', enc)
    for e, c in ext.most_common():
        print('   %-10s n=%5d  bytes=%d' % (e, c, size[e]))
    out[name] = [(i.filename, i.file_size, i.compress_type) for i in infos]
os.makedirs(os.path.join(WORK, 'interim'), exist_ok=True)
json.dump(out, open(os.path.join(WORK, 'interim', 'zip_listing.json'), 'w', encoding='utf-8'), ensure_ascii=False)
for name, lst in out.items():
    print('#####', name)
    shown = 0
    for fn, sz, ct in lst:
        if shown < 60:
            print('  %12d  %s' % (sz, fn))
            shown += 1
