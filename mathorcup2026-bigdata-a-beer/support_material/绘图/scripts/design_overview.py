from repair_common import *
import pandas as pd
from matplotlib.patches import Rectangle,Ellipse,FancyArrowPatch,Polygon
from matplotlib.colors import LinearSegmentedColormap,Normalize
from scipy.stats import gaussian_kde

def arrow(ax,a,b,color=GRAY,lw=.65):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=6,lw=lw,color=color))
def mat(ax,x,y,w,h,c,rows=5,cols=6):
 for i in range(rows):
  for j in range(cols):
   ax.add_patch(Rectangle((x+j*w/cols,y+i*h/rows),w/cols*.88,h/rows*.85,fc=c,alpha=.12+.68*((i*3+j*5)%11)/10,ec='none'))
def overview():
 from design_framework_final import main
 main()

def polar_text(ax,theta,r,text,fs=5.2,color=INK):
 a=np.degrees(theta)%360;rot=90-a
 if 180<a<360:rot+=180;ha='right'
 else:ha='left'
 ax.text(theta,r,text,rotation=rot,rotation_mode='anchor',ha=ha,va='center',fontsize=fs,color=color,clip_on=False)
def geometry(n=50):
 # Start at top, proceed clockwise; three explicit group gaps.
 gap=.10;w=(2*np.pi-3*gap)/n;theta=[];offset=.055
 for lo,hi in [(0,15),(15,30),(30,50)]:
  theta.extend(offset+(np.arange(hi-lo)+.5)*w);offset+=(hi-lo)*w+gap
 return np.array(theta),w

def measurement():
 p=DATA/'fig02_all50_grade_spectra.csv';mp=DATA/'fig02_style_measurement_points.csv';d=pd.read_csv(p);m=pd.read_csv(mp);targets=d.target.drop_duplicates().tolist();labels=d.drop_duplicates('target').set_index('target').label
 f=figure(2);f.set_size_inches(6.7,3.9);ax=f.add_axes([.01,.09,.56,.85],projection='polar');ax.set_theta_zero_location('N');ax.set_theta_direction(-1);ax.set_ylim(0,1.55);ax.axis('off');th,w=geometry();left=np.repeat(.46,50)
 cols=['#E0E8EC','#BBD5D4','#8DBBB6','#5F9F99','#7287A6','#A18BA8','#C97772']
 for k,c in enumerate(cols):
  a=d[d.score==k].set_index('target').loc[targets].proportion.to_numpy();ax.bar(th,a*.60,bottom=left,width=w*.91,color=c,linewidth=0);left+=a*.60
 np.testing.assert_allclose(left,1.06)
 for j,(lo,hi,t)in enumerate([(0,15,'香气'),(15,30,'风味'),(30,50,'口感及扩展')]):
  ax.bar((th[lo]+th[hi-1])/2,.018,bottom=1.105,width=th[hi-1]-th[lo]+w,color=GROUP_COLORS[j],ec='none')
  for i in range(lo,hi):polar_text(ax,th[i],1.18,labels[targets[i]],5.7,GROUP_COLORS[j])
 ax.text(0,0.08,'2197',ha='center',va='center',fontsize=16,color=INK);ax.text(np.pi,.13,'原始品评',ha='center',va='center',fontsize=6.8,color=SECONDARY);ax.text(np.pi,.31,'每柱合计 100%',ha='center',va='center',fontsize=5.4,color=SECONDARY)
 f.text(.035,.962,'a  50维七档原分组成',fontsize=7.2)
 for x,t,c in zip([.045,.205,.365],['香气 15','风味 15','口感及扩展 20'],GROUP_COLORS):f.text(x,.91,t,fontsize=5.8,color=c)
 for k,c in enumerate(cols):f.patches.append(Rectangle((.075+k*.055,.054),.018,.012,transform=f.transFigure,fc=c,ec='none'));f.text(.097+k*.055,.060,str(k),va='center',fontsize=5.9)
 # Exact count-frequency heatmap, aligned to style sample-size track.
 styles=m.groupby('style').size().sort_values(ascending=False).index.tolist();counts=np.arange(3,32);a=np.array([[(m[m['style']==s].records==n).sum()for n in counts]for s in styles]);assert a.sum()==250
 bx=f.add_axes([.735,.21,.205,.685]);cm=LinearSegmentedColormap.from_list('measure',['#FFFFFF','#A4CCCA','#3D817E']);im=bx.imshow(a,cmap=cm,aspect='auto',interpolation='nearest',vmin=0,vmax=max(1,a.max()))
 short={'West Flanders ale':'West Flanders','Flanders old brown':'Old brown','Pils/Lager':'Pils / Lager'}
 bx.set(yticks=range(22),yticklabels=[short.get(s,s)for s in styles],xticks=[0,5,13,21,28],xticklabels=[3,8,16,24,31],xlabel='每款品评次数');bx.tick_params(axis='y',length=0,labelsize=5.4,pad=3);bx.tick_params(axis='x',length=2,labelsize=5.7)
 for s in bx.spines.values():s.set_visible(False)
 for i in np.arange(.5,22):bx.axhline(i,color='#F0F3F4',lw=.3)
 ns=m.groupby('style').size().loc[styles];tx=f.add_axes([.95,.21,.035,.685]);tx.set(xlim=(0,1),ylim=(21.5,-.5));tx.axis('off')
 for i,n in enumerate(ns):tx.text(.5,i,str(n),ha='center',va='center',fontsize=5.4,color=SECONDARY)
 f.text(.624,.962,'b  22风格测量覆盖',fontsize=7.2);f.text(.966,.919,'n',ha='center',fontsize=5.7,color=SECONDARY)
 cb=f.colorbar(im,cax=f.add_axes([.735,.042,.205,.012]),orientation='horizontal');cb.ax.xaxis.set_ticks_position('top');cb.ax.tick_params(length=1,labelsize=5.4,pad=2);cb.outline.set_visible(False);f.text(.735,.012,'相同次数的啤酒数',fontsize=5.6,color=SECONDARY)
 save(f,2,'50维公开原分七档组成与风格测量覆盖。a各径向柱为同一属性2197次品评的0–6档占比，内径相同、柱长总和均为100%；三个分区按原始属性顺序排列，外侧分类弧分别为香气、风味、口感及扩展，不作逐项独立缩放。b精确展示22风格×3–31次品评的啤酒计数，右侧n为各风格啤酒数；全部250款、2197条记录均保留。',[p,mp],{'rating_records':2197,'attributes':50,'n_beers':250,'style_count':22})

def fusion():
 p=DATA/'fig03_per_attribute_branches_fusion.csv';bp=DATA/'fig03_exact_historical_bootstrap_replay.csv';ci=BUNDLE/'research/prediction/fusion_replay_v1/all_prespecified_comparisons.csv';d=pd.read_csv(p);boot=pd.read_csv(bp);comparisons=pd.read_csv(ci)
 A='raw231_y16_no_residual';B='raw231_y50_native';F='inner_groupwise_y16_native50';H='fixed_half_y16_native50';methods=[A,B,F]
 f=figure(3);f.set_size_inches(6.7,3.8);ax=f.add_axes([.005,.06,.58,.91],projection='polar');ax.set_theta_zero_location('N');ax.set_theta_direction(-1);ax.set_ylim(0,1.55);ax.axis('off');th,w=geometry();cm=LinearSegmentedColormap.from_list('performance',['#F4F6F8','#BCD4D3','#3D9690','#2D566D']);norm=Normalize(-.05,.8)
 for k,method in enumerate(methods):ax.bar(th,.145,bottom=.52+k*.17,width=w*.94,color=cm(norm(d[method].to_numpy())),ec='white',lw=.2)
 # Track keys anchor to the gap immediately preceding attribute1.
 for k,name in enumerate(['Y16','Y50','融合']):ax.text(0,.585+k*.17,name,ha='center',va='center',fontsize=5.2,color=INK,bbox=dict(fc='white',ec='none',pad=.15))
 for j,(lo,hi,name)in enumerate([(0,15,'香气'),(15,30,'风味'),(30,50,'口感及扩展')]):
  ax.bar((th[lo]+th[hi-1])/2,.02,bottom=1.065,width=th[hi-1]-th[lo]+w,color=GROUP_COLORS[j],ec='none')
  for i in range(lo,hi):polar_text(ax,th[i],1.13,d.label.iloc[i],5.7,GROUP_COLORS[j])
 ax.text(0,0.05,'50',ha='center',va='center',fontsize=17);ax.text(np.pi,.13,'逐属性 R²',ha='center',va='center',fontsize=6.5,color=SECONDARY)
 f.text(.03,.96,'a  双表示与分组融合图谱',fontsize=7.2)
 for x,t,c in zip([.045,.205,.365],['香气 15','风味 15','口感及扩展 20'],GROUP_COLORS):f.text(x,.91,t,fontsize=5.8,color=c)
 sm=plt.cm.ScalarMappable(norm=norm,cmap=cm);cb=f.colorbar(sm,cax=f.add_axes([.13,.052,.30,.012]),orientation='horizontal',ticks=[0,.4,.8]);cb.ax.tick_params(length=1,labelsize=5.7);cb.outline.set_visible(False)
 bx=f.add_axes([.685,.26,.285,.57]);grid=np.linspace(-.005,.028,350)
 specs=[(F,A,'core30','核心30 · 分组',TEAL),(H,A,'core30','核心30 · 等权',GRAY),(F,B,'all50','全部50 · 分组',TEAL),(H,B,'all50','全部50 · 等权',GRAY)]
 for k,(m,ref,g,name,col)in enumerate(specs):
  y=3-k;vals=boot[(boot.method==m)&(boot.reference==ref)&(boot.targets==g)].delta_r2.to_numpy();assert len(vals)==2000;r=comparisons[(comparisons.method==m)&(comparisons.reference==ref)&(comparisons.targets==g)].iloc[0]
  den=gaussian_kde(vals)(grid);den=den/den.max()*.32;bx.fill_between(grid*100,y+.03,y+.03+den,color=col,alpha=.25,lw=0);bx.plot(np.array([r.conditional_ci025,r.conditional_ci975])*100,[y,y],c=col,lw=1.3);bx.scatter(r.delta_r2*100,y,s=12,c=col,ec='white',lw=.35,zorder=4)
  bx.text(-.65,y+.12,name,ha='right',va='center',fontsize=5.7,color=SECONDARY)
 bx.axvline(0,c='#8C969F',lw=.5,ls='--');bx.set(xlim=(-.55,2.8),ylim=(-.35,3.55),yticks=[],xticks=[0,1,2],xlabel='ΔR² / 百分点');bx.xaxis.labelpad=4;clean(bx,'x');f.text(.625,.96,'b  配对重采样的融合增益',fontsize=7.2);f.text(.685,.125,'点：原估计    线：95%条件区间',fontsize=5.9,color=SECONDARY);f.text(.685,.075,'密度：原2000次配对抽样',fontsize=5.9,color=SECONDARY)
 save(f,3,'50属性共享索引的双表示及融合性能图谱。a三个同心轨道从内至外为Y16、Y50和分组融合的逐属性折外R²，色标固定为−0.05–0.8，三类属性按原顺序分区；完整数值见底表及补图S3。b展示原2000次啤酒整行配对抽样分布、冻结点估计和95%条件区间；抽样严格按原种子复演并与冻结区间一致，未增加模型拟合或新抽样方案。',[p,bp,ci],{'attributes':50,'historical_bootstrap_replays':2000,'new_model_fits':0})
if __name__=='__main__':overview();measurement();fusion()
