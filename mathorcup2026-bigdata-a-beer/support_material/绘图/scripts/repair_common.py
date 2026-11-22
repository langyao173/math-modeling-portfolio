from pathlib import Path
import os,json,hashlib,subprocess
import numpy as np
import fitz
import matplotlib.pyplot as plt
from publication_style import *
BUNDLE=BASE/'recovered/A题_原分补充研究_20261025'
if not (BUNDLE/'research').is_dir():BUNDLE=BASE.parent
OLD=BUNDLE/'chart_rebuild'
OUT=BASE/'figures';DATA=BASE/'plot_data'
OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
SIZES={1:(6.7,3.5),2:(6.7,4.2),3:(6.7,4.0),4:(6.7,3.5),5:(6.7,4.2),6:(6.7,4.0),7:(6.7,3.8),8:(6.7,3.6),9:(6.7,4.4),10:(6.7,3.8),11:(6.7,3.5),12:(6.7,4.0)}
def figure(n):
    setup();return plt.figure(figsize=SIZES[n])
def save(fig,number,caption,sources,extra=None):
    stem=f'fig{number:02d}'
    # Small evidence matrices are vector cells; all marks remain vector paths.
    for ax in fig.axes:
        for im in list(ax.images):
            arr=im.get_array()
            if arr.ndim==2 and arr.size<=5000:
                extent=im.get_extent();nr,nc=arr.shape;xl=ax.get_xlim();yl=ax.get_ylim()
                yedges=np.linspace(extent[3],extent[2],nr+1) if im.origin=='upper' else np.linspace(extent[2],extent[3],nr+1)
                ax.pcolormesh(np.linspace(extent[0],extent[1],nc+1),yedges,arr,cmap=im.get_cmap(),norm=im.norm,shading='flat',edgecolors='none',zorder=im.get_zorder())
                im.remove();ax.set_xlim(xl);ax.set_ylim(yl)
    for ax in fig.axes:
        if ax.get_label()=='<colorbar>':
            for mesh in ax.collections:
                mesh.set_edgecolor('face');mesh.set_linewidth(.12)
    for artist in fig.findobj():
        if hasattr(artist,'set_rasterized'):artist.set_rasterized(False)
    fig.canvas.draw();fig.savefig(OUT/f'{stem}.svg',facecolor='white')
    temp=OUT/f'{stem}.export.pdf'
    subprocess.run(['inkscape','--app-id-tag=reconstruct'+stem,str(OUT/f'{stem}.svg'),'--export-type=pdf','--export-area-page','--export-filename='+str(temp)],capture_output=True,check=True)
    temp.replace(OUT/f'{stem}.pdf')
    with fitz.open(OUT/f'{stem}.pdf')as d:d[0].get_pixmap(dpi=450,alpha=False).save(OUT/f'{stem}.png')
    records=[]
    for item in sources:
        p=Path(item).resolve();records.append(dict(path=os.path.relpath(p,BASE),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    meta=dict(figure=number,caption=caption,figure_inches=list(fig.get_size_inches()),sources=records,new_model_fits=0,new_bootstrap_resamples=0,style='shared_index_compositions_v4')
    if extra:meta.update(extra)
    (OUT/f'{stem}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));plt.close(fig)
