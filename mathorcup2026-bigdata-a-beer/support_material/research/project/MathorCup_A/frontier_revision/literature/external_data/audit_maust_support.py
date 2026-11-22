"""Read-only identity and granularity audit of an external publisher workbook.

The table was inspected with Artifact Tool. This script independently compares
its extracted numbers with the unchanged XLSX cell records. No experiment is
added to the 250-beer analysis and no cells are edited.
"""
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'Maust_2025_fs5c00291_si_001.xlsx'
NS = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def column_index(cell):
    val = 0
    for char in cell:
        if not char.isalpha():
            break
        val = 26 * val + ord(char) - 64
    return val - 1


def main():
    values = json.loads((ROOT/'Maust_2025_Table_S5_values.json').read_text())
    with zipfile.ZipFile(SOURCE) as archive:
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        names = [s.attrib['name'] for s in workbook.findall('s:sheets/s:sheet',NS)]
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']:r.attrib['Target'] for r in rels}
        sheet = [s for s in workbook.findall('s:sheets/s:sheet',NS) if s.attrib['name']=='Table S5'][0]
        key = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
        target = targets[sheet.attrib[key]]
        target = target.lstrip('/') if target.startswith('/') else 'xl/'+target
        xml = ET.fromstring(archive.read(target))
        numbers = []
        n_formula = 0
        errors = []
        for cell in xml.findall('.//s:c',NS):
            if cell.find('s:f',NS) is not None:
                n_formula += 1
            v = cell.find('s:v',NS)
            if v is None or cell.attrib.get('t','n') not in ('n',):
                continue
            addr = cell.attrib['r']
            row = int(''.join(c for c in addr if c.isdigit())) - 1
            col = column_index(addr)
            original = float(v.text)
            got = values[row][col]
            assert isinstance(got,(float,int))
            errors.append(abs(original-got))
            numbers.append(original)
    labels = [row[1] for row in values[5:]]
    assert len(labels)==24 and len(set(labels))==24
    codes = sorted({row[2] for row in values[5:]})
    assert len(codes)==12 and set(row[3] for row in values[5:])=={'4.5','9'}
    assert len(values[4])==199 and len(values[4][30:])==169
    assert max(errors,default=0)<1e-12
    numeric_data = [v for row in values[5:] for v in row[4:] if isinstance(v,(int,float))]
    negatives = [v for v in numeric_data if v<0]
    result = dict(completed=True,source_unmodified=True,
        original_filename='fs5c00291_si_001.xlsx',
        publisher_doi='10.1021/acsfoodscitech.5c00291',
        supporting_doi='10.1021/acsfoodscitech.5c00291.s001',
        source_url='https://acs.figshare.com/articles/dataset/28801756',
        canonical_download_url='https://ndownloader.figshare.com/files/53687869',
        sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),bytes=SOURCE.stat().st_size,
        downloaded_date='2026-10-24',supporting_license='CC BY-NC 4.0 (publisher Figshare record)',
        sheets=names,table_S5=dict(used_rows=29,used_columns=199,data_rows=24,
            labelled_fields=198,descriptor_fields=3,sensory_or_likeness_fields=23,
            physicochemical_fields=3,volatile_fields=169,unique_trial_labels=24,
            yeast_condition_codes=codes,distinct_yeast_strains_from_article=11,
            wort_levels_Plato=[4.5,9],numeric_cells_checked=len(errors),
            max_numeric_extraction_error=max(errors,default=0),formula_cells=n_formula,
            negative_data_entries=len(negatives),minimum_data_value=min(numeric_data)),
        validation_scope='Read-only workbook structure and numerical extraction identity; not a replication of the external study or a new prediction experiment.',
        comparison_boundary='Processed significant-term table from 24 trial beers with different sensory terminology, fermentation conditions and chemical measurement protocols; not appended to the competition training data.',
        notes=['Table S5 combines sensory/beer-likeness means, three physicochemical metrics and volatile columns.',
            'The 12 yeast-condition codes include the same I strain under cold/hot mashing; they are not 12 distinct yeasts.',
            'Sensory repeats and 10 retained panelists are not independent brewing batches.',
            'The author marks targeted MRM columns with green formatting; the extracted value matrix alone does not encode that distinction.',
            'No clipping or repair of near-zero negative numbers, no imputation, no label harmonization, and no workbook export were performed.'])
    (ROOT/'Maust_2025_source_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    report=f'''# 新增公开过程数据的身份核查

原始文件来自 Maust、Sen 与 Lafontaine 的2025年论文支持材料，DOI [10.1021/acsfoodscitech.5c00291.s001](https://doi.org/10.1021/acsfoodscitech.5c00291.s001)。出版方 Figshare 文件为 {SOURCE.stat().st_size:,} 字节，SHA-256为 `{result['sha256']}`。已下载并保留原文件，读取过程中没有重新导出或修改工作簿。支持材料许可按出版方页面记录为CC BY-NC 4.0。

## 数据确实提供了什么

六张表分别为目录及S1–S5。S1列分析标准、内部标准和校准/LOD/LOQ信息；S2列面板术语与训练参照；S3为感官ANOVA；S4为筛选后相关矩阵。S5的有效区域为29×199，前5行是说明/字段信息，后24行对应24种试验啤酒。除一列空白外，包含3个描述字段、23个感官或啤酒相似性结果、3个理化字段和169个挥发物字段。

S5有12个酵母—条件代码，每个分别给4.5和9°P麦汁水平；I-CM与I-HM属于同一酵母的两档糖化条件。因此论文中的11株酵母与表中24行并不矛盾。原文称感官作三次重复，最终面板10人；它们没有被计算为更多独立试酿。方法并未明确给每种条件独立酿造重复，本核查不补写这一事实。

## 为什么暂不拼接为“更多训练样本”

S5明确是经过显著性筛选后用于多因子分析的处理表，感官术语与竞赛50目标不同。MRM定量列通过绿色格式标记，其他挥发物测量需要对照原方法再判断单位和可比性。该表适合后续工艺—化学关系与领域偏移研究；它不是已经对齐的外部50维预测基准。

Artifact Tool读取后，另由原始XLSX XML独立复核 {len(errors)} 个数值单元，最大差为 {max(errors,default=0):.3g}，S5公式单元数为 {n_formula}。记录中的近零负数原样保留；最小值为 {min(numeric_data):.3g}，没有擅自截断。原工作簿、读取程序、值矩阵与本核查JSON一同保存。

另外，Myncke等2025年9种NABLAB酵母加2种参照、每种2批的研究公开了作者接收稿，但其[Zenodo更新版](https://zenodo.org/records/17145184)仍标记文件受限，原始个体感官也未公开。因此没有把“存在DOI”误记成“已获得原始数据”。
'''
    (ROOT/'README.md').write_text(report)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
