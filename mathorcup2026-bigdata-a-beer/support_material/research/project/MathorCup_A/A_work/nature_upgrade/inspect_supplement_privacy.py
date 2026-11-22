# Team review and independent work must be documented truthfully.
"""Read-only source inspection. Never emits person-level source values."""
import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile
import openpyxl

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
path = Path(args.source)
wb = openpyxl.load_workbook(path, data_only=False, read_only=False)
validation = [f'Validation Results {i}' for i in (1, 2, 3)]
names = set()
for name in validation:
    ws = wb[name]
    names.update(ws.cell(r, 2).value for r in range(2, ws.max_row + 1) if ws.cell(r, 2).value is not None)
assert all(isinstance(n, str) for n in names)
ordered = sorted(names)
mapping = {n: f'P{i:04d}' for i, n in enumerate(ordered, 1)}
rows = []
target_data = {}
personal_headers = {'name', 'ip address', 'browser', 'location', 'unique id', 'email', 'email address', 'phone', 'address', 'participant', 'participant name', 'first name', 'last name'}
for ws in wb:
    fields = []
    same_name_cells = []
    ip_cells = []
    email_cells = []
    formula_count = 0
    comments = 0
    nonblank = 0
    for row in ws:
        for cell in row:
            v = cell.value
            if v is None:
                continue
            nonblank += 1
            if cell.data_type == 'f':
                formula_count += 1
            if cell.comment:
                comments += 1
            if isinstance(v, str):
                if cell.row <= 5 and v.lower().strip() in personal_headers:
                    fields.append({'cell': cell.coordinate, 'field': v.lower().strip()})
                if v in names and ws.title not in validation:
                    same_name_cells.append(cell.coordinate)
                if re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}', v.strip()):
                    ip_cells.append(cell.coordinate)
                if re.search(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', v, re.I):
                    email_cells.append(cell.coordinate)
    rows.append({'sheet': ws.title, 'rows': ws.max_row, 'columns': ws.max_column,
                 'nonblank_cells': nonblank, 'formulas': formula_count, 'comments': comments,
                 'personal_header_candidates': fields,
                 'known_participant_exact_matches_outside_validation': same_name_cells,
                 'ip_pattern_cells': ip_cells, 'email_pattern_cells': email_cells})
    if ws.title in validation:
        assert [ws.cell(1, c).value for c in range(1, 17)] == [
            'Time', 'Name', 'Beer', 'Malt', 'Sweetness', 'Hops', 'Bitterness',
            'Esters', 'Fullness (body)', 'Alcohol (warming)', 'Overall appreciation',
            'Comments', 'Browser', 'IP Address', 'Unique ID', 'Location']
        values = []
        for r in range(2, ws.max_row + 1):
            t = ws.cell(r, 1).value
            day = str(t)[:10]
            dt.date.fromisoformat(day)
            values.append([day, mapping[ws.cell(r, 2).value]])
        target_data[ws.title] = {'rows': ws.max_row, 'date_and_anon_name': values}
with ZipFile(path) as z:
    members = z.namelist()
    metadata = {'core_xml_present': 'docProps/core.xml' in members,
                'custom_properties_present': 'docProps/custom.xml' in members,
                'comments_parts': len([p for p in members if re.search(r'/comments\d*\.xml$', p)]),
                'external_link_parts': len([p for p in members if p.startswith('xl/externalLinks/') and p.endswith('.xml')])}
result = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'source_bytes': path.stat().st_size,
          'unique_participant_ids': len(ordered),
          'anonymization': 'Global Python lexical sort of original unmodified strings; IDs P0001, P0002, ...; no mapping exported.',
          'sheets': rows, 'package_metadata_inventory': metadata,
          'sanitized_patch': target_data}
Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'sanitized_patch'}, indent=2))
