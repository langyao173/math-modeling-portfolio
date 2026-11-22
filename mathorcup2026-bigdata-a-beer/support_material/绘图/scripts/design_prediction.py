"""Frozen prediction results, rebuilt as shared-geometry compositions.
"""
from repair_common import *
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
SCEN=['complete','derived_only_missing','MCAR_10_strict','MCAR_30_strict','MCAR_50_strict']
LABEL=['完整','派生','10%','30%','50%']
METHODS=[('tabicl_raw231_y16','TabICL-Y16',TEAL),('old_knn','KNN集成',BLUE),('old_median','中位数集成',PURPLE)]

def fig4():
    p=BUNDLE/'visual_redesign/source_data/historical_plot_data/05_modern_missingness_all.csv'
    d=pd.read_csv(p);d=d[d.targets.eq('core30')]
    cases=['original']+SCEN[1:]
    f=figure(4);f.set_size_inches(6.7,3.25)
    # Identical polar geometry with explicitly truncated, metric-specific
    # baselines. Every bar retains its original value, and every local ruler
    # shows the entire displayed range rather than implying a zero baseline.
    specs=[('r2','宏 R² ↑',.10,.26,1,[.10,.18,.26],2),('grade_mae','等级误差\nMAE ↓',1.40,1.66,1,[1.40,1.53,1.66],2),('grade_within_one','±1档率 ↑',45,60,100,[45,50,55,60],0)]
    records=[]
    for k,(metric,title,minimum,maximum,mult,ticks,digits)in enumerate(specs):
        x=.025+k*.329
        ax=f.add_axes([x,.235,.31,.64],projection='polar')
        ax.set_theta_zero_location('N');ax.set_theta_direction(-1)
        inner=.34;extent=.72
        starts=np.arange(5)*2*np.pi/5
        step=2*np.pi/5;gap=.16;barw=(step-gap)/3
        for j,(m,label,col)in enumerate(METHODS):
            vals=d[d.model.eq(m)].set_index('scenario').loc[cases,metric].to_numpy()*mult
            assert np.all((vals >= minimum) & (vals <= maximum)), (metric, vals)
            theta=starts+gap/2+(j+.5)*barw
            ax.bar(theta,(vals-minimum)/(maximum-minimum)*extent,width=barw*.88,bottom=inner,color=col,edgecolor=INK,linewidth=.32,zorder=3)
            for i,v in enumerate(vals):records.append(dict(method=m,scenario=cases[i],metric=metric,value=float(v),radius_original_zero=False,scale_minimum=minimum,scale_maximum=maximum))
        for t in ticks:
            th=np.linspace(0,2*np.pi,301);ax.plot(th,np.full_like(th,inner+(t-minimum)/(maximum-minimum)*extent),color=GRID,lw=.4,zorder=1)
        # Group sectors, short scenario labels and one consistent reading order.
        for i,a in enumerate(starts):
            th=np.linspace(a+gap/2,a+step-gap/2,50)
            ax.plot(th,np.full_like(th,1.145),color='#A7B2BA',lw=.6,zorder=2)
            center=a+step/2
            ax.text(center,1.25,LABEL[i],ha='center',va='center',fontsize=6.1,color=INK)
        ax.add_artist(Circle((0,0),inner,transform=ax.transData._b,fc='white',ec='#8D99A3',lw=.5,zorder=6))
        ax.text(0,0,title,ha='center',va='center',fontsize=7.3,zorder=7)
        ax.set_ylim(0,1.32);ax.set_xticks([]);ax.set_yticks([]);ax.spines['polar'].set_visible(False);ax.grid(False)
        # A straight local ruler is easier to read than tiny numbers on arcs.
        ruler=f.add_axes([x+.055,.115,.20,.001]);ruler.set_xlim(minimum,maximum);ruler.set_ylim(0,1)
        ruler.set_xticks(ticks);ruler.set_xticklabels([f'{t:.{digits}f}'+('%'if mult==100 else'')for t in ticks],fontsize=5.6)
        ruler.spines[['top','left','right']].set_visible(False);ruler.spines['bottom'].set_color('#A7B2BA');ruler.spines['bottom'].set_linewidth(.45);ruler.set_yticks([]);ruler.tick_params(length=2,width=.4,pad=2)
        f.text(x+.155,.16,f'半径尺度：{minimum:.{digits}f}–{maximum:.{digits}f}'+('%'if mult==100 else''),ha='center',fontsize=5.8,color=SECONDARY)
        f.text(x+.01,.90,'abc'[k],fontsize=8,fontweight='bold',color=INK)
    f.legend(handles=[Line2D([],[],marker='s',ms=5,c=col,ls='',label=lab)for _,lab,col in METHODS],loc='lower center',bbox_to_anchor=(.50,.00),ncol=3,frameon=False,columnspacing=2.1,handletextpad=.5,fontsize=6.4)
    pd.DataFrame(records).to_csv(DATA/'fig04_radial_original_values.csv',index=False)
    save(f,4,'核心30维缺测压力测试。三个同构环形单元分别展示宏R²、相对等级MAE与±1档率；每圈依次为完整、屏蔽派生、10%、30%、50%五种输入场景，每组内均按TabICL-Y16、KNN集成、中位数集成排列。颜色只编码方法。半径采用显式非零截断基线，量程分别为0.10–0.26、1.40–1.66与45%–60%，对应下方各自的比例尺，不能跨圈比较长度；截断只增强指标内差异的可见性，不改变原始数值。数据为冻结五折汇总45值，不表示逐属性或重复训练分布。后四场景先屏蔽5项派生量。',[p],{'aggregate_values':45,'shared_scenario_order':LABEL,'original_zero_radii':False,'radial_scale_ranges':[[.10,.26],[1.40,1.66],[45,60]],'per_attribute_official_missingness_cache_available':False})
    return records


def fig5():
    root=BUNDLE/'raw_score_research';pp=root/'models/results_v2/predictions.npz';tp=root/'models/inputs/raw_panel_means.csv';mp=root/'models/results_v2/raw_metrics.csv';cip=root/'protocol/independent_audit_v2/paired_contrasts_for_report.csv'
    z=np.load(pp,allow_pickle=False);truth=pd.read_csv(tp).set_index('beer').loc[z['beer'],z['targets']].to_numpy()[:,:30]
    per=pd.read_csv(DATA/'fig05_per_attribute_missingness.csv');beers=pd.read_csv(DATA/'fig05_paired_beer_mse_differences.csv');metric=pd.read_csv(mp)
    metric=metric[metric.target_weighting.eq('record_equal')&metric.targets.eq('core30')]
    checks=[]
    for method in ['direct_ridge','cdf_et','cdf_ridge']:
        for sc in SCEN:
            pred=z[f'point__{method}__{sc}'][:,:30];value=np.sqrt(np.mean((pred-truth)**2));old=metric[metric.method.eq(method)&metric.scenario.eq(sc)].iloc[0]
            assert abs(value-old.pooled_mean_rmse)<1e-12
            checks.append(dict(method=method,scenario=sc,rmse=float(value)))
    f=figure(5);f.set_size_inches(6.7,4.2)
    # A single tightly fitted joint unit, inspired by regression / marginal /
    # residual compounds. All 7,500 frozen beer-target cells remain visible.
    main=[.125,.415,.323,.515];ax=f.add_axes(main);top=f.add_axes([.125,.937,.323,.035],sharex=ax);right=f.add_axes([.454,.415,.026,.515],sharey=ax)
    res=f.add_axes([.045,.415,.060,.515],sharey=ax);res_hist=f.add_axes([.045,.937,.060,.035],sharex=res)
    obs=truth.ravel();pred=z['point__cdf_et__complete'][:,:30].ravel();err=pred-obs
    seq=LinearSegmentedColormap.from_list('prediction_density',['#F5F8F8','#B9D5D5',TEAL,'#1A5357'])
    ax.hexbin(obs,pred,gridsize=40,mincnt=1,bins='log',cmap=seq,linewidths=0,rasterized=True)
    ax.plot([0,6],[0,6],c=INK,lw=.65,ls=(0,(3,2)),zorder=3)
    # Binned conditional mean and distribution spread are descriptive, not CIs.
    bx=[];by=[];qlo=[];qhi=[]
    for l in np.arange(0,6,.5):
        ix=(obs>=l)&(obs<l+.5 if l<5.5 else obs<=6)
        if ix.sum()<15:continue
        bx.append(float(obs[ix].mean()));by.append(float(pred[ix].mean()));qlo.append(float(np.quantile(pred[ix],.1)));qhi.append(float(np.quantile(pred[ix],.9)))
    ax.fill_between(bx,qlo,qhi,color=RED,alpha=.14,lw=0,zorder=2);ax.plot(bx,by,c=RED,lw=.85,zorder=4)
    ax.set(xlim=(0,6),ylim=(0,6),xticks=[0,2,4,6],yticks=[0,2,4,6],xlabel='观测原分均值',ylabel='');clean(ax);f.text(.012,.68,'CDF-ET 预测均值',rotation=90,ha='center',va='center',fontsize=6.5)
    ax.text(.04,.96,'250 × 30\nRMSE 0.685',transform=ax.transAxes,va='top',fontsize=6.2,color=INK)
    # Residual and prediction share the same vertical coordinate exactly.
    res.hexbin(err,pred,gridsize=(10,35),mincnt=1,bins='log',cmap=seq,linewidths=0,rasterized=True);res.axvline(0,c=RED,lw=.6)
    res.set(xlim=(-5,3),xticks=[-4,0,3],xlabel='残差');res.tick_params(axis='x',labelsize=5.4,pad=2);res.tick_params(axis='y',left=False,labelleft=False);clean(res);res.spines['left'].set_visible(False)
    for sub,v,col,orient,bins in [(top,obs,BLUE,'vertical',np.linspace(0,6,25)),(right,pred,TEAL,'horizontal',np.linspace(0,6,25)),(res_hist,err,TEAL,'vertical',np.linspace(-5,3,20))]:
        sub.hist(v,bins=bins,orientation=orient,color=col,alpha=.6,ec='white',lw=.25);sub.axis('off')
    f.text(.045,.998,'a  原分校准',fontsize=7.2,va='top')
    # Paired beer errors, in a consistently aligned right-side evidence column.
    b=f.add_axes([.655,.735,.315,.20]);b.axvline(0,c=INK,lw=.5,ls=(0,(3,2)))
    cols={'direct_ridge':BLUE,'cdf_et':TEAL,'z_affine':PURPLE};labs={'direct_ridge':'Ridge','cdf_et':'CDF-ET','z_affine':'仿射标定'}
    for yy,method in enumerate(cols):
        a=beers[beers.method.eq(method)].mse_difference.to_numpy();jit=.16*np.sin(np.arange(len(a))*2.399963229728653)
        b.scatter(a,yy+jit,c=cols[method],s=3.5,alpha=.32,lw=0,rasterized=True)
        q=np.quantile(a,[.25,.5,.75]);b.plot(q[[0,2]],[yy,yy],c=cols[method],lw=2.2);b.scatter(q[1],yy,c=cols[method],s=11,ec='white',lw=.3,zorder=5)
    b.set_xscale('symlog',linthresh=.2,linscale=.7);b.set(xlim=(-2.15,8.6),ylim=(2.5,-.5),xticks=[-2,-.2,0,.2,2,8],xticklabels=['−2','−.2','0','.2','2','8'],yticks=[0,1,2],yticklabels=list(labs.values()))
    clean(b,'x');b.tick_params(axis='y',length=0,labelsize=6.1);b.tick_params(axis='x',labelsize=5.5);b.set_xlabel('逐啤酒 ΔMSE（对称log）',fontsize=5.8,labelpad=2)
    f.text(.57,.998,'b  相对训练分布参照',fontsize=7.2,va='top')
    # Frozen aggregate paired intervals remain numerically complete.
    ci=pd.read_csv(cip);sel=ci[ci.target_weighting.eq('record_equal')&ci.targets.eq('core30')&ci.scenario.eq('complete')&ci.reference.eq('training_distribution')]
    f.text(.585,.638,'汇总差值与95%条件区间',fontsize=6.3,color=INK)
    for i,(method,label,col)in enumerate([('cdf_et','CDF-ET',TEAL),('z_affine','仿射标定',PURPLE)]):
        r=sel[sel.method.eq(method)&sel.metric.eq('pooled_mean_rmse')].iloc[0]
        f.text(.585,.600-i*.041,label,fontsize=6.1,color=col)
        f.text(.97,.600-i*.041,f'ΔRMSE {r.delta:+.3f}  [{r.ci025:+.3f}, {r.ci975:+.3f}]',ha='right',fontsize=5.8,color=col)
    r=sel[sel.method.eq('cdf_et')&sel.metric.eq('rps')].iloc[0]
    f.text(.585,.506,f'ΔRPS {r.delta:+.4f}  [{r.ci025:+.4f}, {r.ci975:+.4f}]',fontsize=6.0,color=TEAL)
    f.text(.585,.460,'点：250款啤酒   粗条：IQR   中点：中位数',fontsize=5.6,color=SECONDARY)
    # Coupled 30-attribute tracks. Same attribute order and scenario rows;
    # signed differences encode improvement, leaving exact values to source data.
    cmap=LinearSegmentedColormap.from_list('delta_error',[BLUE,'#FFFFFF',RED])
    mats=[];target_order=list(z['targets'][:30])
    for xx,metriccol,ref,title,limit in [(.085,'target_rmse','direct_ridge','c  ΔRMSE：CDF-ET − Ridge',.18),(.585,'target_rps','cdf_ridge','d  ΔRPS：CDF-ET − CDF-Ridge',.009)]:
        arr=[]
        for sc in SCEN:
            sub=per[per.scenario.eq(sc)].set_index(['method','target']);a=sub.loc['cdf_et'].reindex(target_order)[metriccol].to_numpy()-sub.loc[ref].reindex(target_order)[metriccol].to_numpy();arr.append(a)
        arr=np.array(arr);limit=max(limit,float(np.max(np.abs(arr))));mats.append(arr)
        c=f.add_axes([xx,.120,.35,.15]);im=c.imshow(arr,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto',interpolation='nearest')
        c.set(xticks=[0,14,29],xticklabels=['1','15','30'],yticks=range(5),yticklabels=LABEL if xx<.5 else []);c.tick_params(length=0,labelsize=5.6,pad=3)
        for spine in c.spines.values():spine.set_visible(False)
        c.axvline(14.5,c='white',lw=1.2);c.set_xticks(np.arange(-.5,30),minor=True);c.set_yticks(np.arange(-.5,5),minor=True);c.grid(which='minor',c='white',lw=.25);c.tick_params(which='minor',length=0)
        f.text(xx,.322,title,fontsize=6.8)
        f.text(xx+.0875,.284,'香气 15',fontsize=5.8,color=SECONDARY,ha='center');f.text(xx+.2625,.284,'风味 15',fontsize=5.8,color=SECONDARY,ha='center')
        cb=f.colorbar(im,cax=f.add_axes([xx,.072,.35,.014]),orientation='horizontal');cb.set_ticks([-limit,0,limit]);cb.ax.set_xticklabels([f'{-limit:.3f}','0',f'{limit:.3f}']);cb.ax.tick_params(labelsize=5.3,length=1.5,pad=2);cb.outline.set_visible(False)
    f.text(.50,.012,'冷蓝：误差降低   ·   30属性顺序固定，五行对应输入场景',ha='center',fontsize=5.9,color=SECONDARY)
    assert all(a.shape==(5,30)for a in mats)
    save(f,5,'原分校准与缺测误差结构。a以固定比例拼合250款×30属性的观测—预测联合密度、真实边际频数和残差条；虚线为理想校准，珊瑚线与浅带为观测分箱后的预测均值及10%—90%分位，非置信区间。b为每模型250款啤酒相对训练分布参照的MSE差，使用对称log轴保留极端值，粗条为IQR；原汇总RMSE与RPS差的既有95%条件配对区间完整列出。c、d共用30属性顺序与五输入场景，分别展示CDF-ET相对Ridge的逐属性RMSE差、相对CDF-Ridge的逐属性RPS差，负值表示更小误差；色标各自独立，完整数值与原始误差保存底表。所有7500单元不作为独立实验重复，区间仍以啤酒配对。',[pp,tp,mp,cip,DATA/'fig05_per_attribute_missingness.csv',DATA/'fig05_paired_beer_mse_differences.csv'],{'calibration_cells':7500,'beer_difference_points':750,'missingness_delta_cells':300,'raw_metric_rechecks':checks,'new_intervals':False})
    return checks

if __name__=='__main__':
    rows=fig4();checks=fig5()
    (BASE/'prediction_design_audit.json').write_text(json.dumps(dict(status='PASS',fig4_original_values=45,fig4_polar_baselines=[.10,1.40,45],fig4_radial_scales=[[.10,.26],[1.40,1.66],[45,60]],fig5_calibration_cells=7500,fig5_beer_points=750,fig5_missingness_delta_cells=300,raw_metric_checks=checks,new_model_fits=0,new_bootstrap_resamples=0),ensure_ascii=False,indent=2))
    print('Prediction compositions exported and frozen values checked.')
