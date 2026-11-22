# Team review and independent work must be documented truthfully.
"""Verify an anonymized copy without exporting source identifiers.

Uses openpyxl only to read. Workbook authoring is performed separately by
the Artifact Tool builder. This file intentionally never writes a name map.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import openpyxl

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--copy', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--expected-source-sha256', default=None)
args = parser.parse_args()
source = Path(args.source); copy = Path(args.copy)
if args.expected_source_sha256:
    assert sha(source) == args.expected_source_sha256, 'source_hash_changed'
old = openpyxl.load_workbook(source, data_only=False)
new = openpyxl.load_workbook(copy, data_only=False)
validation = [f'Validation Results {i}' for i in (1, 2, 3)]
assert old.sheetnames == new.sheetnames
names = sorted({old[s].cell(r, 2).value for s in validation for r in range(2, old[s].max_row + 1)})
mapping = {name: f'P{i:04d}' for i, name in enumerate(names, 1)}
assert list(mapping.values()) == sorted(mapping.values())
checks = []
removed_strings = {'Name': set(names), 'IP Address': set(), 'Unique ID': set(), 'Browser': set(), 'Location': set()}
for s in validation:
    a, b = old[s], new[s]
    for row in a.iter_rows(min_row=2):
        for index, field in [(12, 'Browser'), (13, 'IP Address'), (14, 'Unique ID'), (15, 'Location')]:
            v = row[index].value
            if isinstance(v, str) and len(v) >= 8:
                removed_strings[field].add(v)
    for r in range(2, a.max_row + 1):
        original_date = str(a.cell(r, 1).value)[:10]
        actual = b.cell(r, 1).value
        assert isinstance(actual, dt.datetime) and actual.time() == dt.time(), (s, 'date_type_or_time')
        assert str(actual)[:10] == original_date, (s, 'date_changed')
        assert b.cell(r, 2).value == mapping[a.cell(r, 2).value], (s, 'name_mapping')
        for c in range(3, 12):
            assert b.cell(r, c).value == a.cell(r, c).value, (s, 'sensory_value_changed', r, c)
    assert [b.cell(1, c).value for c in range(1, 12)] == [a.cell(1, c).value for c in range(1, 12)]
    assert all(b.cell(r, c).value is None for r in range(1, max(a.max_row, b.max_row)+1) for c in range(12, max(a.max_column, b.max_column)+1))
    checks.append({'sheet': s, 'records': a.max_row-1, 'retained_columns': 11,
                   'dates_preserved_without_time': True, 'stable_lexical_name_order': True,
                   'beer_and_eight_scores_exact': True, 'removed_columns_empty': True})

preserved = []
global_max_diff = 0.0
for s in old.sheetnames:
    if s in validation:
        continue
    a, b = old[s], new[s]
    numeric_roundtrip_count = 0
    maximum_diff = 0.0
    formula_count = 0
    hard_differences = []
    for row in a:
        for cell in row:
            x, y = cell.value, b[cell.coordinate].value
            if cell.data_type == 'f':
                formula_count += 1
            if x == y:
                continue
            if isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)) and not isinstance(y, bool):
                diff = abs(x-y)
                if math.isclose(x, y, rel_tol=1e-14, abs_tol=1e-14):
                    numeric_roundtrip_count += 1
                    maximum_diff = max(maximum_diff, diff)
                    continue
            if (x is None and y == '') or (x == '' and y is None):
                continue
            hard_differences.append(cell.coordinate)
    a_nonblank = {c.coordinate for row in a for c in row if c.value is not None}
    extra = [c.coordinate for row in b for c in row if c.value is not None and c.coordinate not in a_nonblank]
    assert not hard_differences, (s, 'unrelated_cell_value_or_formula_changes', hard_differences[:20], len(hard_differences))
    assert not extra, (s, 'unrelated_new_cells', extra[:20])
    assert {str(x) for x in a.merged_cells.ranges} == {str(x) for x in b.merged_cells.ranges}, (s, 'merged_ranges')
    global_max_diff = max(global_max_diff, maximum_diff)
    preserved.append({'sheet': s, 'nonblank_source_cells': len(a_nonblank), 'formula_count': formula_count,
                      'formula_text_exact': True, 'non_numeric_values_exact': True,
                      'numeric_not_bitwise_equal_count': numeric_roundtrip_count, 'maximum_numeric_absolute_difference': maximum_diff,
                      'merged_ranges_preserved': True, 'sheet_state_preserved': a.sheet_state == b.sheet_state,
                      'freeze_panes_preserved': a.freeze_panes == b.freeze_panes,
                      'tables_source_copy': [len(a.tables), len(b.tables)],
                      'charts_source_copy': [len(a._charts), len(b._charts)],
                      'images_source_copy': [len(a._images), len(b._images)],
                      'validations_source_copy': [len(a.data_validations.dataValidation), len(b.data_validations.dataValidation)],
                      'conditional_format_rules_source_copy': [len(a.conditional_formatting), len(b.conditional_formatting)]})

with ZipFile(copy) as z:
    all_text_nodes = []
    for part in z.namelist():
        if not part.endswith('.xml'):
            continue
        root = ET.fromstring(z.read(part))
        all_text_nodes.extend(e.text for e in root.iter() if e.text)
    remaining = {field: sum(v in set(all_text_nodes) for v in values) for field, values in removed_strings.items()}
    assert not any(remaining.values()), ('removed_identifier_values_retained_in_xml', remaining)
    meta = ET.fromstring(z.read('docProps/core.xml')) if 'docProps/core.xml' in z.namelist() else None
    metadata_person_fields = {e.tag.split('}')[-1]: bool(e.text) for e in meta or [] if e.tag.split('}')[-1] in ['creator','lastModifiedBy']}
    assert not any(metadata_person_fields.values()), ('personal_core_metadata_present', metadata_person_fields)

output = {'input_sha256': sha(source), 'output_sha256': sha(copy),
          'input_bytes': source.stat().st_size, 'output_bytes': copy.stat().st_size,
          'sheet_names_and_order_preserved': True, 'sheet_count': len(old.sheetnames),
          'unique_participant_identifiers': len(names),
          'name_map': 'Not persisted. Anonymous IDs P0001..P0025 follow lexical order of original name strings.',
          'validation_checks': checks, 'unchanged_scientific_sheets': preserved,
          'maximum_scientific_numeric_roundtrip_absolute_difference': global_max_diff,
          'removed_source_identifier_values_remaining_in_xml': remaining,
          'nonempty_core_person_metadata_fields': metadata_person_fields,
          'original_source_file_unchanged': sha(source) == args.expected_source_sha256 if args.expected_source_sha256 else None}
Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in output.items() if k not in ['unchanged_scientific_sheets']},indent=2))
