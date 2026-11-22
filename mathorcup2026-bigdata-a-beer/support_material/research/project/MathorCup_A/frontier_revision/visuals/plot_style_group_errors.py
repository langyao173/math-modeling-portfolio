"""Editable-vector style-group NMSE comparison from verified Table A2 data.

Two panels retain original attachment style names and a common numerical axis.
No nested-selection point or legend entry is drawn unless all 22 nested values
are present and finite. This program performs no model fitting or resampling.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mathorcup_a1_matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_COLUMNS = (
    ('six_model_nmse', '六模型', 'slate', 'o'),
    ('fixed_y16_nmse', '固定 Y16', 'teal', 'D'),
    ('nested_selection_nmse', '训练内选模', 'orange', '^'),
)
REQUIRED = {'style', 'n', 'six_model_nmse', 'fixed_y16_nmse', 'nested_selection_nmse'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate_project():
    for ancestor in (HERE, *HERE.parents):
        for candidate in (ancestor, ancestor / 'project/MathorCup_A'):
            if ((candidate / 'data/A/1.csv').is_file()
                    or (candidate / 'A_work/q1/run_q1.py').is_file()):
                return candidate
    raise FileNotFoundError('Supply --project-dir for the original or prepared source project.')


def default_font(project):
    bundled = project / 'assets/fonts/NotoSansCJKsc-Regular.otf'
    if bundled.is_file():
        return bundled
    try:
        return Path(font_manager.findfont(font_manager.FontProperties(family='Noto Sans CJK SC'),
                                         fallback_to_default=False))
    except ValueError as error:
        raise FileNotFoundError('Install Noto Sans CJK SC or supply its file with --font-path.') from error


def configure(style_path, font_path):
    # Reuse the existing style palette; local sizing is tuned to a 16.2 cm page.
    # Do not call its configure/save helpers, which write into that source tree.
    spec = importlib.util.spec_from_file_location('existing_publication_style', style_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    palette = module.COLORS.copy()
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        'font.family': family, 'font.size': 7.8,
        'axes.labelcolor': palette['ink'], 'text.color': palette['ink'],
        'xtick.color': palette['text'], 'ytick.color': palette['text'],
        'axes.edgecolor': palette['line'], 'axes.linewidth': .55,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.unicode_minus': False,
        'xtick.major.size': 2.5, 'xtick.major.width': .55,
        'ytick.major.size': 0, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'svg.fonttype': 'none',
        'svg.hashsalt': 'mathorcup-a-source-backed-figure-A1',
        'mathtext.fontset': 'dejavusans',
    })
    return palette, family


def read_table(path, require_nested):
    frame = pd.read_csv(path)
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f'Table A2 requires columns {sorted(REQUIRED)}.')
    if len(frame) != 22 or not frame['style'].is_unique or frame['style'].isna().any():
        raise ValueError('Exactly 22 unique, named original styles are required.')
    if not np.isfinite(frame['n']).all() or not np.equal(frame['n'], frame['n'].astype(int)).all():
        raise ValueError('Sample counts must be finite integers.')
    frame['n'] = frame['n'].astype(int)
    if frame['n'].sum() != 250 or (frame['n'] <= 0).any():
        raise ValueError('Style counts must partition the original 250 beers.')
    if (frame['n'] >= 10).sum() != 13 or (frame['n'] < 10).sum() != 9:
        raise ValueError('The original styles partition into 13 and 9 groups at n=10.')
    for column in ('six_model_nmse', 'fixed_y16_nmse'):
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f'Incomplete fixed-model error values: {column}')
    nested = frame['nested_selection_nmse']
    nested_present = bool(nested.notna().any())
    if nested_present and (not np.isfinite(nested).all() or (nested < 0).any()):
        raise ValueError('A partially filled nested column must not be plotted.')
    if require_nested and not nested_present:
        raise ValueError('--require-nested refuses an uncomputed nested-selection column.')
    return frame, nested_present


def safe_id(text):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text))


def build_figure(frame, models, palette):
    width_cm, height_cm = 16.2, 9.9
    fig = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54), dpi=120)
    panel_specs = [
        ('A', frame['n'] >= 10, .196, .270, .013, .176, '样本量 ≥ 10'),
        ('B', frame['n'] < 10, .706, .274, .495, .686, '样本量 < 10'),
    ]
    maximum = float(frame[[model[0] for model in models]].max().max())
    xmax = max(2.2, math.ceil((maximum + .08) / .2) * .2)
    ticks = np.arange(0., xmax + 1e-9, .5)
    ordered, label_boxes = [], []
    offsets = [-.16, .16] if len(models) == 2 else [-.22, 0., .22]
    for letter, mask, axleft, axwidth, nameleft, nright, title in panel_specs:
        group = frame.loc[mask].sort_values(['six_model_nmse', 'style'], ascending=[False, True], kind='stable')
        ax = fig.add_axes([axleft, .155, axwidth, .635])
        ax.set_gid(f'A1_panel_{letter}')
        ax.set(xlim=(0., xmax), ylim=(len(group) - .60, -.60), yticks=[], xticks=ticks)
        ax.set_xticklabels([f'{value:g}' for value in ticks], fontsize=7.2)
        ax.tick_params(axis='x', pad=3)
        ax.set_axisbelow(True)
        for tick in ticks:
            if not np.isclose(tick, 1.):
                ax.axvline(tick, color=palette['line'], lw=.45, alpha=.65, zorder=0)
        reference = ax.axvline(1., color=palette['slate'], lw=.8, ls=(0, (3, 3)), zorder=1)
        reference.set_gid(f'A1_{letter}_overall_reference_E1')
        label_x = (nameleft - axleft) / axwidth
        count_x = (nright - axleft) / axwidth
        for row_number, row in enumerate(group.itertuples(index=False)):
            row_map = row._asdict()
            xs = [float(row_map[column]) for column, *_ in models]
            ys = [row_number + offset for offset in offsets]
            ax.axhline(row_number, color=palette['line'], lw=.35, alpha=.30, zorder=0)
            line, = ax.plot(xs, ys, color=palette['gray'], lw=.8, solid_capstyle='round', zorder=2)
            line.set_gid(f'A1_{letter}_{safe_id(row.style)}_paired_estimates')
            for (column, label, color_key, marker), value, y in zip(models, xs, ys):
                scatter = ax.scatter([value], [y], s=13.5 if marker == 'o' else 14.5,
                                     marker=marker, edgecolors=palette[color_key], linewidths=.85,
                                     facecolors='white' if marker == 'o' else palette[color_key], zorder=4)
                scatter.set_gid(f'A1_{letter}_{safe_id(row.style)}_{column}')
            label = ax.text(label_x, row_number, row.style, transform=ax.get_yaxis_transform(),
                            fontsize=7.7, va='center', ha='left', clip_on=False)
            count = ax.text(count_x, row_number, str(row.n), transform=ax.get_yaxis_transform(),
                            fontsize=7.4, color=palette['text'], va='center', ha='right', clip_on=False)
            label.set_gid(f'A1_{letter}_{safe_id(row.style)}_label')
            count.set_gid(f'A1_{letter}_{safe_id(row.style)}_n')
            label_boxes.append((label, count, ax))
            ordered.append({'panel': letter, 'style': row.style, 'n': int(row.n),
                            **{column: float(row_map[column]) for column, *_ in models}})
        fig.text(nameleft, .857, letter, fontsize=10.5, weight='bold', color=palette['orange'])
        fig.text(nameleft + .036, .858, title, fontsize=8.8)
        fig.text(nameleft, .812, '风格', fontsize=7.6, color=palette['text'])
        fig.text(nright, .812, r'$n$', fontsize=8., ha='right', color=palette['text'])
    handles = [Line2D([], [], color=palette[color_key], marker=marker, lw=0,
                      markersize=4.2, markeredgewidth=.85,
                      markerfacecolor='white' if marker == 'o' else palette[color_key], label=label)
               for _, label, color_key, marker in models]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .992),
               ncol=len(models), fontsize=8.4, handlelength=1.0,
               columnspacing=2.1, handletextpad=.55, borderaxespad=0.)
    fig.text(.5, .070, '核心感官 NMSE（越低越好）', fontsize=8.2, ha='center')
    fig.text(.5, .018, 'E = 1：全样本均值预测器的总体参考误差',
             fontsize=7., color=palette['text'], ha='center')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_check = []
    for label, count, ax in label_boxes:
        first = label.get_window_extent(renderer)
        second = count.get_window_extent(renderer)
        if first.x1 + 1 > second.x0 or second.x1 + 1 > ax.get_window_extent(renderer).x0:
            raise RuntimeError(f'Label/count collision at {label.get_text()!r}.')
        label_check.append({'style': label.get_text(), 'label_to_n_gap_px': float(second.x0 - first.x1),
                            'n_to_axes_gap_px': float(ax.get_window_extent(renderer).x0 - second.x1)})
    outside = []
    canvas = fig.bbox
    for text in fig.findobj(match=lambda artist: isinstance(artist, Text)):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -.1 or box.y0 < -.1 or box.x1 > canvas.width + .1 or box.y1 > canvas.height + .1:
            outside.append(text.get_text())
    if outside:
        raise RuntimeError(f'Text exceeds fixed figure canvas: {outside}')
    return fig, ordered, {'width_cm': width_cm, 'height_cm': height_cm,
                           'shared_xlim': [0., xmax], 'label_checks': label_check,
                           'text_outside_canvas': outside}


def caption(models):
    names = '、'.join(model[1] for model in models)
    return ('图 A1　风格分组的核心感官预测误差。'
            f'比较{names}。A、B 分别展示 n≥10 的 13 类风格和 n<10 的 9 类风格，n 为原始啤酒数。'
            '逐款核心 30 维折外平方误差以各目标在全部 250 款啤酒中的总体方差归一化，组内平均得到风格 NMSE。'
            '各面板内按六模型 NMSE 降序排列，横轴尺度一致；点间连线仅关联同一风格的模型估计。'
            '虚线 E=1 对应全样本均值预测器在全部样本上的总体参考误差，不代表某风格训练均值的误差。'
            '小样本组用于描述；全部结果来自原随机五折划分。')


def add_svg_description(svg_path, description):
    namespace = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', namespace)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(svg_path)
    root = tree.getroot()
    title = ET.Element(f'{{{namespace}}}title', {'id': 'A1_title'})
    title.text = '图 A1 风格分组的核心感官预测误差'
    desc = ET.Element(f'{{{namespace}}}desc', {'id': 'A1_description'})
    desc.text = description
    root.insert(0, title)
    root.insert(1, desc)
    root.set('role', 'img')
    root.set('aria-labelledby', 'A1_title A1_description')
    tree.write(svg_path, encoding='utf-8', xml_declaration=True)
    text_nodes = root.findall(f'.//{{{namespace}}}text')
    if len(text_nodes) < 44 or root.findall(f'.//{{{namespace}}}image'):
        raise RuntimeError('SVG must retain native text and contain no raster image.')
    return len(text_nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path)
    parser.add_argument('--table', type=Path)
    parser.add_argument('--style-path', type=Path)
    parser.add_argument('--font-path', type=Path)
    parser.add_argument('--output-prefix', type=Path)
    parser.add_argument('--require-nested', action='store_true')
    args = parser.parse_args()
    project = (args.project_dir or locate_project()).resolve()
    relative_table = HERE.parent / 'experiments/modern_prediction/generalization_groups/table_A2_core30_nmse.csv'
    if not relative_table.is_file():
        relative_table = project / 'frontier_revision/experiments/modern_prediction/generalization_groups/table_A2_core30_nmse.csv'
    relative_style = HERE / 'publication_style.py'
    if not relative_style.is_file():
        relative_style = project / 'frontier_revision/visuals/publication_style.py'
    table_path = (args.table or relative_table).resolve()
    style_path = (args.style_path or relative_style).resolve()
    font_path = (args.font_path or default_font(project)).resolve()
    output = (args.output_prefix or HERE / 'A1_style_core30_nmse').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame, nested_present = read_table(table_path, args.require_nested)
    models = MODEL_COLUMNS if nested_present else MODEL_COLUMNS[:2]
    palette, family = configure(style_path, font_path)
    fig, ordered, layout = build_figure(frame, models, palette)
    text = caption(models)
    fig.savefig(output.with_suffix('.svg'), metadata={'Creator': 'Python / Matplotlib', 'Date': None})
    fig.savefig(output.with_suffix('.png'), dpi=300)
    plt.close(fig)
    native_text_nodes = add_svg_description(output.with_suffix('.svg'), text)
    output.with_suffix('.caption.txt').write_text(text + '\n', encoding='utf-8')
    pd.DataFrame(ordered).to_csv(output.with_suffix('.plotted_data.csv'), index=False)
    facts = {
        'title': '图 A1 风格分组的核心感官预测误差',
        'input_table': str(table_path), 'input_table_sha256': sha(table_path),
        'drawing_source_sha256': sha(Path(__file__)),
        'existing_style_source': str(style_path), 'existing_style_sha256': sha(style_path),
        'font': family, 'font_sha256': sha(font_path),
        'models_drawn': [label for _, label, _, _ in models],
        'nested_values_present': nested_present,
        'absent_nested_values_were_invented': False,
        'original_beers': 250, 'styles': 22,
        'panel_groups': {'A': {'n_rule': 'n>=10', 'styles': 13, 'beers': 195},
                         'B': {'n_rule': 'n<10', 'styles': 9, 'beers': 55}},
        'metric': 'v_j=(1/250) sum_i (Y_ij-Ybar_j)^2; individual loss=mean_{j=1..30}((Y_ij-pred_ij)^2/v_j); style E_s=mean individual loss in style s.',
        'reference_E1': 'Overall NMSE of the constant full-sample-mean predictor on all 250 rows equals 1. This is not a within-style training-mean error benchmark.',
        'same_random_outer_cv_not_leave_style_out': True,
        'intervals_drawn': False,
        'connecting_lines': 'Connect only models for the same style; not uncertainty intervals or an interpolation over different styles.',
        'sort': 'Within each panel: descending six-model NMSE; style name breaks exact ties. Order never uses nested results.',
        'native_svg_text_nodes': native_text_nodes, 'embedded_raster_images': 0,
        'layout': layout,
        'weighted_core30_nmse': {column: float(np.average(frame[column], weights=frame['n'])) for column, *_ in models},
        'style_equal_core30_nmse': {column: float(frame[column].mean()) for column, *_ in models},
    }
    output.with_suffix('.facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'svg': str(output.with_suffix('.svg')), 'png': str(output.with_suffix('.png')),
                      'models': facts['models_drawn'], 'nested_present': nested_present,
                      'canvas_cm': [layout['width_cm'], layout['height_cm']],
                      'native_svg_text_nodes': native_text_nodes}, ensure_ascii=False))


if __name__ == '__main__':
    main()
