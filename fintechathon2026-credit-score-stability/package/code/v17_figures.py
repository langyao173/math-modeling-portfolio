"""Publication revision: same observations, quieter annotation, explicit denominators."""
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';FONT=ROOT/'paper/fonts/ReportChartCJK.ttf'
font_manager.fontManager.addfont(str(FONT));fam=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family':fam,'font.size':9,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'text.color':'#34412E','axes.labelcolor':'#34412E','xtick.color':'#728069','ytick.color':'#485840','grid.color':'#E7EBE1','grid.linewidth':.5})
SAGE='#587B68';CLAY='#B58864';GRAY='#B3BEB5';INK='#2D3C33';W=156/25.4
used=[]
def read(p): used.append(ROOT/p);return pd.read_csv(ROOT/p)
def save(f,n):
 for ext in ['pdf','svg','png']:f.savefig(OUT/f'v17_{n}.{ext}',dpi=350,facecolor='white')
 plt.close(f)
def mechanism():
 c=read('external/zenodo/cohort_summary.csv').set_index('cohort');f=plt.figure(figsize=(W,2.6))
 a=f.add_axes([.19,.27,.31,.55]);names=['all_published_customers','publisher_test_customers','first_observed_positive_limit_increase']
 for y,k in zip([2,1,0],names):
  r=c.loc[k];v=r.masking_share*100;a.errorbar(v,y,xerr=[[v-r.cluster_ci95_low*100],[r.cluster_ci95_high*100-v]],fmt='o',color=SAGE,capsize=3,ms=4,lw=1.2)
  a.text(v+2,y+.06,f'{v:.2f}%',va='center',fontsize=9,color=INK)
 a.set(xlim=(0,65),ylim=(-.5,2.5),xticks=[0,20,40,60],yticks=[2,1,0],yticklabels=['全部配对','留出客户','每客首次提额'],xlabel='占余额上升配对的比例（%）');a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False);a.grid(axis='x');a.set_axisbelow(True)
 f.text(.035,.94,'a   余额增加，使用率仍下降',size=10,weight='bold')
 a=f.add_axes([.74,.27,.21,.55]);r=c.loc[names[0]];v=100*r.reference_util_over_one/r.increase_transitions
 a.barh([1,0],[0,v],height=.34,color=['#C3CCBA',CLAY]);a.plot(0,1,'o',ms=3,color=GRAY)
 a.text(.7,1,'0.00%',va='center',size=9);a.text(v+.7,0,f'{v:.2f}%',va='center',size=9,color=CLAY)
 a.set(xlim=(0,26),ylim=(-.6,1.6),yticks=[1,0],yticklabels=['当期额度','提额前额度'],xticks=[0,10,20],xlabel='余额超过该额度（%）');a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False);a.grid(axis='x');a.set_axisbelow(True)
 f.text(.575,.94,'b   旧尺度下的余额范围',size=10,weight='bold')
 f.text(.035,.035,'a：26,816次余额上升配对；误差线为客户簇95%区间。  b：75,597次提额配对。',size=8,color='#7B8672');save(f,'external_mechanism')
def epoch():
 d=read('results/epoch/epoch_attribution_rows.csv.gz');f=plt.figure(figsize=(W,2.8));a=f.add_axes([.11,.26,.39,.60]);x=d.fixed_reference_update;y=d.stale_offset_update;mask=d.direction_reversed_by_stale_offset.astype(bool);lo=min(x.min(),y.min())-2;hi=max(x.max(),y.max())+2
 a.axhline(0,color='#BAC5B0',lw=.65);a.axvline(0,color='#BAC5B0',lw=.65);a.plot([lo,hi],[lo,hi],ls='--',color='#A3AF9B',lw=.7)
 a.scatter(x[~mask],y[~mask],s=5,c=GRAY,alpha=.55,edgecolors='none',rasterized=False);a.scatter(x[mask],y[mask],s=8,c=CLAY,alpha=.8,edgecolors='none')
 a.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='固定参考：ΔQ（分）',ylabel='固定旧偏移：ΔQ（分）');a.set_aspect('equal',adjustable='box');a.grid(alpha=.5)
 f.text(.035,.94,'a   同一信息更新的两种路径',size=10,weight='bold')
 a=f.add_axes([.70,.26,.25,.60]);v=d.absolute_index_discrepancy;qs=[.5,.9,.95,.99];vals=v.quantile(qs).values
 for yy,label,val in zip([3,2,1,0],['P50','P90','P95','P99'],vals):
  col=CLAY if label=='P95' else SAGE;a.hlines(yy,0,val,color=col,lw=1.2);a.plot(val,yy,'o',ms=4,color=col);a.text(val+.25,yy,f'{val:.2f}',va='center',size=9,color=col)
 a.set(xlim=(0,11),ylim=(-.6,3.6),xticks=[0,5,10],yticks=[3,2,1,0],yticklabels=['P50','P90','P95','P99'],xlabel='绝对路径差（分）');a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False);a.grid(axis='x');a.set_axisbelow(True)
 f.text(.59,.94,'b   参考路径差的分布',size=10,weight='bold')
 f.text(.035,.035,f'2,677对同账户相邻月记录；陶土色标示448对变化方向相反的记录。最大路径差 {v.max():.2f} 分。',size=8,color='#7B8672');save(f,'reference_epoch')
if __name__=='__main__':
 mechanism();epoch()
 import v15_figures as legacy
 legacy.save=save
 legacy.benchmarks();legacy.capacity_appendix();used.extend(legacy.USED)
 (OUT/'v17_figure_sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in used},indent=2))
