#!/usr/bin/env python3
"""Audit the supplied 2026 MathorCup A data without altering its values.


"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path(os.environ.get('MATHORCUP_DATA_DIR', '/workspace/scratch/a7d3fe1f9c53/contest_2026/A/附件')))
    ap.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    src, out = args.source, args.output
    out.mkdir(parents=True, exist_ok=True)
    frames = {str(i): pd.read_csv(src / f'{i}.csv') for i in (1, 2, 4, 5, 6, 7)}
    x, y, text = frames['1'], frames['4'], frames['7']
    xi, yi = x.set_index('beer'), y.set_index('beer')
    chem_cols, sensory_cols = list(x.columns[3:]), list(y.columns[3:])
    c2, c5, c6 = (frames[k].set_index(frames[k].columns[0]) for k in ('2', '5', '6'))
    th = pd.read_excel(src/'8.xlsx', sheet_name='Sensory thresholds')
    sn = pd.read_excel(src/'8.xlsx', sheet_name='Sensory names')
    cal = pd.read_excel(src/'8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    th['field_name_R'] = th['field_name_R'].astype(str).str.strip()
    th_lookup = th.drop_duplicates('field_name_R').set_index('field_name_R')
    th.to_csv(out/'thresholds_source.csv', index=False)
    sn.to_csv(out/'sensory_names_source.csv', index=False)
    cal.to_csv(out/'calibrated_chemistry_source.csv', index=True)
    workbook = openpyxl.load_workbook(src/'8.xlsx', read_only=True, data_only=True)
    inventory=[]
    for sheet in workbook:
        header=next(sheet.values, ())
        inventory.append(dict(workbook='8.xlsx',sheet=sheet.title,rows_including_header=sheet.max_row,
                              columns=sheet.max_column,headers=json.dumps(list(header),ensure_ascii=False)))
    pd.DataFrame(inventory).to_csv(out/'workbook_inventory.csv',index=False)
    evidence=[]
    for address in ['B45','F45','B55','G55','J55','K55','B67','C67','B127','G127','J127','K127',
                    'B146','G146','J146','K146','B155','G155','J155','K155']:
        evidence.append(dict(workbook='8.xlsx',sheet='Sensory thresholds',cell=address,
                             value=workbook['Sensory thresholds'][address].value))
    pd.DataFrame(evidence).to_csv(out/'key_cell_evidence.csv',index=False)
    workbook.close()

    basic = {}
    for key, df in frames.items():
        basic[key] = {'rows': len(df), 'columns': len(df.columns), 'missing_total': int(df.isna().sum().sum()),
                      'missing_by_column': df.isna().sum()[lambda z:z>0].to_dict()}
        if 'beer' in df:
            basic[key].update(beer_unique=int(df.beer.nunique()), beer_min=int(df.beer.min()),
                              beer_max=int(df.beer.max()), beer_id_unique=int(df.beer_id.nunique()),
                              duplicated_rows=int(df.duplicated().sum()))

    # Correlation S5 puts carbonation before aftertaste and body, unlike S4.
    sensory_matrix_order = sensory_cols[:46] + ['co2', 'aftertaste', 'body', 'overall']
    corr_report = {}
    matrix_geometry = {}
    for key, matrix in [('2',c2),('5',c5)]:
        values=matrix.to_numpy(float)
        eigenvalues=np.linalg.eigvalsh(values)
        matrix_geometry[key]={'symmetry_max_error':float(abs(values-values.T).max()),
                              'diagonal_max_error':float(abs(np.diag(values)-1).max()),
                              'minimum_eigenvalue':float(eigenvalues.min()),
                              'negative_eigenvalue_n':int((eigenvalues < -1e-8).sum())}
    for method in ('spearman', 'pearson'):
        matrices = {
            '2': (x[chem_cols].corr(method=method).values, c2.values),
            '5': (y[sensory_matrix_order].corr(method=method).values, c5.values),
        }
        name_to_chem = dict(zip(c2.index, chem_cols))
        chemical6 = [name_to_chem[n] for n in c6.index]
        joint = pd.concat([xi[chemical6], yi[sensory_matrix_order]], axis=1)
        recomputed6 = joint.corr(method=method).loc[chemical6, sensory_matrix_order].values
        matrices['6'] = (recomputed6, c6.values)
        for key, (calc, provided) in matrices.items():
            delta = calc - provided
            corr_report[f'{key}_{method}'] = dict(max_abs=float(np.nanmax(abs(delta))),
                                                 mean_abs=float(np.nanmean(abs(delta))),
                                                 rmse=float(np.sqrt(np.nanmean(delta**2))))

    # Known raw positive concentrations make the first 37 transformations identifiable.
    first37 = chem_cols[:37]
    transform_rows, mismatch_rows = [], []
    for col in cal.columns[2:]:
        a = pd.to_numeric(xi[col], errors='coerce')
        b = pd.to_numeric(cal[col], errors='coerce').reindex(a.index)
        finite = a.notna() & b.notna()
        positive = a.notna() & b.gt(0)
        mode = 'log10' if col in first37 else 'raw_scale_with_some_cross_sheet_differences'
        pred = np.log10(b.where(b.gt(0))) if col in first37 else b
        err = (a-pred).abs()
        matches = err.le(1e-7) & err.notna()
        disagree = err.gt(1e-7) & err.notna()
        minpos = float(b[b.gt(0)].min())
        floor = np.log10(minpos/2)
        badraw = b.le(0) | b.isna()
        row = dict(field=col, transform=mode, positive_raw_n=int(positive.sum()),
                   paired_n=int(finite.sum()), match_n=int(matches.sum()), mismatch_n=int(disagree.sum()),
                   median_identity_error=float((a-b).abs().median()),
                   median_log10_error=float((a[positive]-np.log10(b[positive])).abs().median()),
                   median_ln_error=float((a[positive]-np.log(b[positive])).abs().median()),
                   max_comparison_error=float(err.max()), zero_raw_n=int(b.eq(0).sum()),
                   negative_raw_n=int(b.lt(0).sum()), missing_raw_n=int(b.isna().sum()),
                   missing_file1_n=int(a.isna().sum()), half_min_positive=minpos/2,
                   replaced_nonpositive_or_missing_n=int(badraw.sum()) if col in first37 else None,
                   max_floor_error=float((a[badraw]-floor).abs().max()) if col in first37 and badraw.any() else None)
        transform_rows.append(row)
        for beer in a.index[disagree]:
            mismatch_rows.append(dict(beer=int(beer), field=col, supplied1=float(a[beer]),
                                      calibrated8=float(b[beer]), expected_transform=float(pred[beer]),
                                      absolute_difference=float(err[beer])))
    transforms = pd.DataFrame(transform_rows)
    transforms.to_csv(out/'transformation_evidence.csv', index=False)
    pd.DataFrame(mismatch_rows).to_csv(out/'cross_sheet_value_differences.csv', index=False)

    chemical_meta = []
    for i, field in enumerate(chem_cols):
        t = th_lookup.loc[field]
        transform = 'verified_log10' if field in first37 else (
            'raw_calibrated_scale_with_exceptions' if field in cal.columns else (
                'derived_index_scale_unverified' if field in chem_cols[-4:] else 'supplied_abundance_scale'))
        row = dict(index_file1=i, field=field, matrix2_name=c2.index[i], transform=transform,
                   calibrated_sheet_available=field in cal.columns, is_derived=field in chem_cols[-4:]+['kcalperc'],
                   internal_standard=t['origin_paper']=='Internal standard solution',
                   chemical_name_conflict=field=='X2.ethylhexylsalicylate')
        row.update({k:t[k] for k in ('chemical_name','chemical_name_short','calibrated','origin_paper','unit','aging','off_flavors','threshold_mean','threshold_median')})
        row.update(missing_n=int(x[field].isna().sum()), negative_n=int(x[field].lt(0).sum()),
                   unique_nonmissing=int(x[field].nunique()), minimum=float(x[field].min()), maximum=float(x[field].max()))
        chemical_meta.append(row)
    pd.DataFrame(chemical_meta).to_csv(out/'chemical_metadata.csv', index=False)

    sensory_meta = []
    sn_panel = sn.loc[sn.dataset.eq('Trained panel')].set_index('R_name')
    display_map = dict(zip(sensory_matrix_order, c5.index))
    for i, field in enumerate(sensory_cols):
        t = sn_panel.loc[field]
        sensory_meta.append(dict(index_file4=i, field=field, display=display_map[field],
                                 group=t['Category_Paper'], origin=t['Category'],
                                 annotation_type=t['DataType'], has_AF_prefix=field.startswith(('A_', 'F_')),
                                 mean=float(y[field].mean()), std=float(y[field].std()),
                                 minimum=float(y[field].min()), maximum=float(y[field].max()),
                                 raw_0_6_available=False))
    pd.DataFrame(sensory_meta).to_csv(out/'sensory_metadata.csv', index=False)

    wf = openpyxl.load_workbook(src/'3.xlsx', data_only=False)
    wv = openpyxl.load_workbook(src/'3.xlsx', data_only=True)
    template_report = []
    formula_rows = []
    for sheet in wf:
        if not sheet.title.startswith('beer_'):
            continue
        formulas = [c for row in sheet for c in row if c.data_type=='f']
        refs = sorted(set(r for c in formulas for r in re.findall(r'ISBLANK\(([A-Z]+[0-9]+)\)', str(c.value))))
        nonempty = [r for r in refs if sheet[r].value is not None]
        template_report.append(dict(sheet=sheet.title, printed_beer=sheet['S3'].value,
                                    formula_n=len(formulas), blank_check_input_n=len(refs),
                                    nonempty_inputs=nonempty,
                                    cached_nonzero_formula_n=sum(wv[sheet.title][c.coordinate].value not in (None, 0) for c in formulas)))
        for c in formulas:
            formula_rows.append(dict(sheet=sheet.title, cell=c.coordinate, formula=c.value,
                                     cached=wv[sheet.title][c.coordinate].value))
    pd.DataFrame(formula_rows).to_csv(out/'tasting_card_formulas.csv', index=False)

    # External files are source-check evidence; never append these identical observations.
    external = {}
    extpath = out/'external'/'original_supplement.xlsx'
    if extpath.exists():
        book = openpyxl.load_workbook(extpath, data_only=True, read_only=True)
        external['sheets'] = [{'sheet':s.title, 'rows':s.max_row, 'columns':s.max_column} for s in book]
        external['identical_source_checks'] = {}
        for key, sheet in [('1','Supplementary File S1'),('4','Supplementary File S4'),('7','Supplementary File S7')]:
            orig = pd.read_excel(extpath, sheet_name=sheet).set_index('beer')
            provided = frames[key].set_index('beer')
            cols = list(provided.columns[2:])
            a = provided[cols].apply(pd.to_numeric, errors='coerce')
            b = orig.reindex(a.index)[cols].apply(pd.to_numeric, errors='coerce')
            d = (a-b).abs()
            external['identical_source_checks'][key] = dict(max_abs=float(np.nanmax(d)),
                                                           different_over_1e_7=int((d>1e-7).sum().sum()),
                                                           missing_disagreement=int((a.isna()!=b.isna()).sum().sum()))

    energy_fields = ['ethanol..v.v.','GFS','unfermentables','protein.g.L.','lactic_acid.mg.L.']
    energy_data = cal[energy_fields+['kcalperc']].apply(pd.to_numeric, errors='coerce').dropna()
    energy_formula = (5.523*energy_data['ethanol..v.v.']
                      +.4*(energy_data['GFS']+energy_data['unfermentables']+energy_data['protein.g.L.'])
                      +.0003*energy_data['lactic_acid.mg.L.'])
    energy_error = energy_data['kcalperc']-energy_formula
    energy_report = {'source':'8.xlsx / Calibrated chemical dataset', 'n':len(energy_data),
                    'formula':'5.523*ethanol_%v/v+0.4*(GFS_g/L+unfermentables_g/L+protein_g/L)+0.0003*lactic_acid_mg/L',
                    'max_abs_residual':float(abs(energy_error).max()),
                    'rmse':float(np.sqrt(np.mean(energy_error**2))),
                    'file1_kcalperc_transform':'log10'}
    report = dict(basic=basic, beer_key_same_order={key:bool(x.beer.equals(frames[key].beer)) for key in ('4','7')},
                  styles=x.tasting_category_fine.value_counts().to_dict(), correlations=corr_report,
                  correlation_matrix_geometry=matrix_geometry,
                  first37_log10=first37, transformation_rows=transform_rows,
                  card_sheets=wf.sheetnames, template_report=template_report,
                  summary_barnyard_formula=wf['summary']['AX2'].value,
                  chemical_count=231, derived_count=5, base_measurement_count=226,
                  trained_sensory_count=50, aroma_flavour_core_count=30,
                  forum_descriptor_count=186, forum_all_missing_columns=[c for c in text.columns if text[c].isna().all()],
                  original_supplement=external, caloric_formula=energy_report)
    (out/'audit_results.json').write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(clean_json({'output':str(out), 'correlations':corr_report,
                                'metadata_counts':{'chemical':len(chemical_meta),'sensory':len(sensory_meta)},
                                'raw_scale_cross_sheet_mismatches':len(mismatch_rows),
                                'template_nonempty_inputs':sum(len(z['nonempty_inputs']) for z in template_report)}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
