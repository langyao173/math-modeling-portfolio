"""Render the prespecified semantic control results without selecting models."""
from pathlib import Path
import json
import hashlib
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'style_controls'


def main():
    complete = json.loads((OUT / 'COMPLETE.json').read_text())
    assert complete['complete'] and complete['fitted_outer_models'] == 60
    data = pd.read_csv(OUT / 'model_comparison.csv')
    controls = data[(data.stage == 'blend') & data.model.str.startswith('within_style_')
                    & data.model.str.endswith('_all50')]
    style = data[(data.stage == 'blend') & data.model.eq('style_only_all50')].iloc[0]
    style_rrr = data[(data.stage == 'rrr') & data.model.eq('style_only_all50')].iloc[0]
    contrasts = json.loads((OUT / 'contrasts.json').read_text())
    paired = [x for x in contrasts if x['stage'] == 'blend'
              and x['control'].startswith('within_style_') and x['control'].endswith('_all50')]
    comparison = next(x for x in contrasts if x['stage'] == 'blend' and x['control'] == 'style_only_all50')
    assert len(controls) == len(paired) == 5
    lower_positive = sum(x['all50_ci'][0] > 0 for x in paired)
    lo, hi = comparison['all50_ci']
    selection = json.loads((OUT / 'nested_selection.json').read_text())
    ranks = sorted(set(x['spec']['rank'] for x in selection if x['family'] == 'style_only_all50'))
    text = (
        '为分离风格信息与评论配对信息，进一步在每个实际训练分区内保留风格、置换评论，并以五个固定种子分别重复完整嵌套流程。'
        f'按全部50维选参时，五个置换融合的$R^2$为{controls.all50_r2.min():.5f}—{controls.all50_r2.max():.5f}；'
        f'真实对齐相对它们增加{min(x["all50_delta"] for x in paired):.5f}—{max(x["all50_delta"] for x in paired):.5f}，'
        f'{lower_positive}/5个条件配对区间下限为正。这一结果支持配对信息的小幅互补作用。五次控制独立列出，区间未经多重比较校正。\n\n'
        f'同时，仅将训练风格标签作为辅助目标，单分支已达到{style_rrr.all50_r2:.5f}，融合达到{style.all50_r2:.5f}；'
        f'真实评论融合相对它的差值只有{comparison["all50_delta"]:.5f}，区间[{lo:.5f},{hi:.5f}]。'
        '因此，本文将贡献定位于训练期辅助机制；风格标签提供了简洁而有效的替代。'
        f'风格辅助五折选中秩均为{ranks[0] if len(ranks)==1 else ranks}；'
        '它与高维词频任务的辅助空间不同，作为任务替换诊断单独解释。'
    )
    (OUT / 'manuscript_results.md').write_text(text + '\n')
    sources = ['COMPLETE.json', 'model_comparison.csv', 'contrasts.json', 'nested_selection.json']
    provenance = dict(final_predictors_unchanged=True,
        sources={f: hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in sources},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'summary_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(text)


if __name__ == '__main__':
    main()
