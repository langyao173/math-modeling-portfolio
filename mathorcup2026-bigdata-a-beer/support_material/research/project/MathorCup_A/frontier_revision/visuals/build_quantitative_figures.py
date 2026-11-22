#!/usr/bin/env python3
"""Read executed experiment records and emit research figures in SVG/PDF/PNG.

No performance values, uncertainty bounds or network edges are synthesized.
Optional additional tables require the same data split and metric definition.

"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parent/"qa/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator
from publication_style import COLORS as C, configure, panel_title, clean, dot_handle, save

HERE=Path(__file__).resolve().parent
OUT=HERE/"quantitative"
SOURCES={}
PLOT_DATA={}


def read_csv(path):
    path=Path(path)
    d=pd.read_csv(path)
    SOURCES[str(path)]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(d),columns=d.columns.tolist())
    return d


def read_json(path):
    path=Path(path)
    j=json.loads(path.read_text())
    SOURCES[str(path)]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),format="json")
    return j


def export_rows(name,df):
    p=OUT/"data"/f"{name}.csv";p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False)
    PLOT_DATA[name]=dict(path=str(p),rows=len(df),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read_npz(path):
    path=Path(path)
    with np.load(path,allow_pickle=False) as arrays:
        result={key:arrays[key] for key in arrays.files}
    SOURCES[str(path)]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),format="npz",
                          shapes={key:list(value.shape) for key,value in result.items()})
    return result


def load_modern_predictions(experiment,work,paired_model):
    """Activate the modern comparison only after all registered models finish."""
    if experiment is None:return None
    table=experiment/"model_comparison.csv";tabdpt=experiment/"tabdpt"
    required={"rank_pca64_y16_no_residual","rank_pca64_y16_ridge_residual",
              "raw231_y16_no_residual","raw231_y16_ridge_residual","raw231_y50_native"}
    necessary=[table,experiment/"oof_arrays.npz",tabdpt/"model_comparison.csv",
               tabdpt/"oof_arrays.npz",experiment/"paired_bootstrap.csv"]
    if not all(path.is_file() for path in necessary):return None
    if not required<=set(pd.read_csv(table).model):return None
    metrics=read_csv(table).set_index("model")
    dpt_metrics=read_csv(tabdpt/"model_comparison.csv").set_index("model")
    arrays=read_npz(experiment/"oof_arrays.npz")
    dpt_arrays=read_npz(tabdpt/"oof_arrays.npz")
    original=read_npz(work/"q1/oof_arrays.npz")
    folds=read_csv(work/"q1/fold_assignments.csv").set_index("beer")
    for data in [arrays,dpt_arrays]:
        np.testing.assert_array_equal(data["Y"],original["Y"])
        np.testing.assert_array_equal(data["fold"],folds.loc[data["beer"],"fold"])
        assert data["Y"].shape==(250,50) and set(data["fold"])==set(range(5))
    np.testing.assert_array_equal(arrays["beer"],dpt_arrays["beer"])
    np.testing.assert_array_equal(arrays["targets"],dpt_arrays["targets"])
    core=np.array([str(t).startswith(("A_","F_")) for t in arrays["targets"]])
    assert core.sum()==30
    labels={"rank_pca64_y16_no_residual":"TabICL · X64 / Y16",
            "rank_pca64_y16_ridge_residual":"TabICL · X64＋残差",
            "raw231_y16_no_residual":"TabICL · Y16",
            "raw231_y16_ridge_residual":"TabICL · Y16＋残差",
            "raw231_y50_native":"TabICL · 原生 50 维"}
    denominator=((arrays["Y"]-arrays["Y"].mean(axis=0))**2).sum(axis=0)
    rows=[];dimensions=[]
    for key,label in labels.items():
        pred=arrays[key]
        assert pred.shape==(250,50) and np.isfinite(pred).all()
        r2=1-((arrays["Y"]-pred)**2).sum(axis=0)/denominator
        np.testing.assert_allclose([r2.mean(),r2[core].mean()],metrics.loc[key,["macro_r2","core30_r2"]].to_numpy(float),atol=1e-12,rtol=0)
        rows.append(dict(model=key,label=label,macro_r2=r2.mean(),core30_r2=r2[core].mean(),n_beers=250,protocol="outer5_same_split"))
        dimensions.extend(dict(model=key,target=t,r2=r,core30=bool(c)) for t,r,c in zip(arrays["targets"],r2,core))
    assert len(dpt_metrics)==1
    dpt_key=dpt_metrics.index[0];pred=dpt_arrays["prediction"]
    assert pred.shape==(250,50) and np.isfinite(pred).all()
    r2=1-((arrays["Y"]-pred)**2).sum(axis=0)/denominator
    np.testing.assert_allclose([r2.mean(),r2[core].mean()],dpt_metrics.loc[dpt_key,["macro_r2","core30_r2"]].to_numpy(float),atol=1e-12,rtol=0)
    rows.append(dict(model=dpt_key,label="TabDPT v1.3‡",macro_r2=r2.mean(),core30_r2=r2[core].mean(),n_beers=250,protocol="outer5_same_split"))
    dimensions.extend(dict(model=dpt_key,target=t,r2=r,core30=bool(c)) for t,r,c in zip(arrays["targets"],r2,core))
    intervals=read_csv(experiment/"paired_bootstrap.csv")
    if paired_model==dpt_key:
        intervals=pd.concat([intervals,read_csv(tabdpt/"paired_bootstrap.csv")],ignore_index=True)
    selected=intervals[(intervals.model==paired_model)&(intervals.reference=="old_nested_ensemble")].set_index("targets").loc[["all50","core30"]]
    assert len(selected)==2 and (selected.repetitions==2000).all()
    assert (selected.ci025<=selected.ci975).all()
    read_json(experiment/"experiment_spec.json")
    return dict(rows=pd.DataFrame(rows),dimensions=pd.DataFrame(dimensions),intervals=selected,
                paired_model=paired_model,label=labels.get(paired_model,"TabDPT v1.3"))


def prediction(work, extra=None,semantic_dir=None,modern_dir=None,paired_model="raw231_y16_no_residual"):
    d=read_csv(work/"q1/dimension_metrics.csv")
    f=read_csv(work/"nature_upgrade/fusion/dimension_metrics.csv")
    labels={"training_mean":"训练均值", "ridge_raw":"原尺度 Ridge", "style_mean":"已知风格均值†",
            "pls":"PLS", "ridge_rank":"秩尺度 Ridge", "rbf":"RBF",
            "random_forest":"Random Forest", "extra_trees":"Extra Trees", "nested_ensemble":"六模型集成",
            "fusion8_shrink":"八分支研究对照"}
    rows=[]
    for model,g in d.groupby("model",sort=False):
        rows.append(dict(model=model,label=labels[model],macro_r2=g.r2.mean(),
                         core30_r2=g.loc[g.target.str.startswith(("A_","F_")),"r2"].mean(),
                         n_beers=250,protocol="outer5_same_split"))
    g=f[f.model=="fusion8_shrink"]
    rows.append(dict(model="fusion8_shrink",label=labels["fusion8_shrink"],macro_r2=g.r2.mean(),
                     core30_r2=g.loc[g.target.str.startswith(("A_","F_")),"r2"].mean(),n_beers=250,protocol="outer5_same_split"))
    z=pd.DataFrame(rows)
    if semantic_dir is not None:
        sem=read_csv(semantic_dir/"blend_comparison.csv").set_index("model")
        r=sem.loc["nested_blend_aligned_aux_all50"]
        z=pd.concat([z,pd.DataFrame([dict(model="nested_blend_aligned_aux_all50",label="语义辅助融合",
            macro_r2=r.all50_r2,core30_r2=r.core30_r2,n_beers=250,protocol="outer5_same_split")])],ignore_index=True)
    if extra:
        e=read_csv(extra)
        required={"model","label","macro_r2","core30_r2","n_beers","protocol"}
        if not required<=set(e):raise ValueError(f"Prediction table requires {required}")
        if not ((e.n_beers==250)&(e.protocol=="outer5_same_split")).all():
            raise ValueError("Do not mix external or unmatched validation with the original five folds.")
        z=pd.concat([z,e],ignore_index=True)
    modern=load_modern_predictions(modern_dir,work,paired_model)
    if modern is not None:z=pd.concat([z,modern["rows"]],ignore_index=True)
    assert z.model.is_unique
    z=z.sort_values("macro_r2",ascending=False,kind="stable").reset_index(drop=True)
    export_rows("04_prediction_model_summary",z)
    # Full baseline range is retained, including the negative raw-scale Ridge.
    fig=plt.figure(figsize=(8.6,max(5.3,.31*len(z)+1.5)))
    gs=fig.add_gridspec(1,2,width_ratios=[1.40,1],left=.235,right=.98,bottom=.225,top=.82,wspace=.42)
    ax=fig.add_subplot(gs[0]); yy=np.arange(len(z))
    for i,r in z.iterrows():
        ax.plot([r.core30_r2,r.macro_r2],[i,i],color=C["line"],lw=1.4,zorder=1)
    ax.scatter(z.macro_r2,yy,s=32,c=C["teal"],zorder=3)
    ax.scatter(z.core30_r2,yy,s=32,facecolors="white",edgecolors=C["orange"],linewidths=1.3,zorder=4)
    ax.axvline(0,color=C["gray"],lw=.8)
    ax.set(yticks=yy,yticklabels=z.label,xlim=(min(-.61,z[["macro_r2","core30_r2"]].min().min()-.07),max(.35,z[["macro_r2","core30_r2"]].max().max()+.06)),
           ylim=(len(z)-.55,-.65),xlabel="五折汇总宏平均 R²")
    for i,r in z.iterrows():
        ax.annotate(f"{r.macro_r2:.3f}",(r.macro_r2,i),xytext=(5,-2),textcoords="offset points",fontsize=10,va="bottom",color=C["teal"])
    clean(ax,"x");panel_title(ax,"a","同折预测比较")
    ax.legend(handles=[dot_handle(C["teal"],"全部 50 维"),dot_handle(C["orange"],"香气 / 风味 30 维",filled=False)],
              loc="lower left",bbox_to_anchor=(-.01,1.12),ncol=2,columnspacing=1.3,handletextpad=.35)
    if modern is None:
        ax=fig.add_subplot(gs[1])
        one=d[d.model=="extra_trees"].set_index("target")
        two=d[d.model=="nested_ensemble"].set_index("target").loc[one.index]
        reference_name="Extra Trees";candidate_name="六模型集成"
    else:
        position=gs[1].get_position(fig)
        scatter_height=position.width*fig.get_figwidth()/fig.get_figheight()
        ax=fig.add_axes([position.x0,.82-scatter_height,position.width,scatter_height])
        one=d[d.model=="nested_ensemble"].set_index("target")
        two=modern["dimensions"].query("model == @paired_model").set_index("target").loc[one.index]
        reference_name="冻结六模型";candidate_name=modern["label"]
    targets=one.index;core=np.array([t.startswith(("A_","F_")) for t in targets])
    lo=min(one.r2.min(),two.r2.min())-.035;hi=max(one.r2.max(),two.r2.max())+.035
    ax.plot([lo,hi],[lo,hi],color=C["gray"],lw=1,ls=(0,(3,3)),zorder=1)
    ax.scatter(one.r2[~core],two.r2[~core],s=29,facecolors="white",edgecolors=C["gray"],linewidths=1.2,zorder=2,label="其他 20 维")
    ax.scatter(one.r2[core],two.r2[core],s=30,c=C["teal"],alpha=.88,zorder=3,label="香气 / 风味 30 维")
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel=f"{reference_name} · 逐维 R²",ylabel=f"{candidate_name} · 逐维 R²")
    ax.set_aspect("equal",adjustable="box");clean(ax,"both")
    panel_title(ax,"b","50 个目标的逐维比较",y=1.06 if modern is None else 1+.0396/scatter_height)
    ax.legend(loc="lower right",fontsize=9.8)
    wins=int((two.r2>one.r2).sum())
    ax.text(.04,.96,f"{wins} / 50 维提高",transform=ax.transAxes,va="top",fontsize=10,color=C["teal"])
    if modern is None:
        fig.text(.05,.07,"250 款啤酒 · 同一五折；核心任务为香气 / 风味 30 维，右图保留全部 50 个目标。",fontsize=9.5,color=C["text"])
        fig.text(.05,.025,"† 该参照额外接收真实风格标签；其余预测器的部署输入为化学测量。",fontsize=9.5,color=C["text"])
    else:
        cx=fig.add_axes([position.x0,.20,position.width,.15])
        ci=modern["intervals"]
        for i,(group,color) in enumerate([("all50",C["teal"]),("core30",C["orange"])]):
            row=ci.loc[group]
            cx.errorbar(row.delta_r2,i,xerr=[[row.delta_r2-row.ci025],[row.ci975-row.delta_r2]],
                        fmt="o",color=color,ms=5.5,lw=1.7,capsize=3,
                        markerfacecolor=color if group=="all50" else "white",zorder=3)
        cx.axvline(0,color=C["text"],lw=.9)
        margin=max(.004,float(ci.ci975.max()-ci.ci025.min())*.12)
        cx.set(xlim=(min(0,ci.ci025.min())-margin,max(0,ci.ci975.max())+margin),
               ylim=(1.55,-.55),yticks=[0,1],yticklabels=["50 维","30 维"],xlabel="ΔR²（相对冻结六模型）")
        clean(cx,"x");cx.tick_params(axis="y",length=0)
        interval_title="Y16 配对增量" if paired_model=="raw231_y16_no_residual" else f"{candidate_name} 增量"
        panel_title(cx,"c",interval_title,y=1.16)
        export_rows("04_modern_paired_uncertainty",ci.reset_index())
        zero_crossing=(ci.ci025<=0)&(ci.ci975>=0)
        interval_note="两组区间均跨零。" if zero_crossing.all() else ""
        fig.text(.05,.078,"250 款 · 同一五折；c 为固定折外预测的 2000 次配对啤酒重采样。"+interval_note,fontsize=9.5,color=C["text"])
        fig.text(.05,.038,"† 风格均值额外接收真实风格标签。‡ TabDPT 的预训练来源边界见正文。",fontsize=9.5,color=C["text"])
    paired=pd.DataFrame(dict(target=targets,reference_model=reference_name,candidate_model=candidate_name,
                             reference_r2=one.r2.to_numpy(),candidate_r2=two.r2.to_numpy(),core30=core))
    export_rows("04_prediction_paired_dimensions",paired)
    caption=f"同一五折外层验证下比较全部原基准、语义辅助和已完成的现代候选；实心与空心点分别表示 50 维和核心 30 维宏平均 R²。带 † 的风格均值额外使用样本真实风格标签，属于条件参照；其余预测器部署时只接收化学测量。b 逐维配对{candidate_name}与{reference_name}，虚线为相等线，全部 50 个感官目标均保留。"
    if modern is not None:
        caption+=f" c 展示{candidate_name}相对{reference_name}的宏平均 R² 差及 2000 次配对啤酒 bootstrap 95% 百分位区间，每次重算样本目标方差；b、c的候选相同，区间不对应a中其他配置。区间基于固定折外预测，不包含重训波动或多模型选择校正。"
        for group,label in [("all50","全部50维"),("core30","核心30维")]:
            row=ci.loc[group]
            caption+=f"{label}增量为{row.delta_r2:.6f}，区间[{row.ci025:.6f},{row.ci975:.6f}]。"
        caption+="TabDPT 的预训练来源边界单列，不将来源尚不完备的模型视为完全清洁的独立外部验证。"
    save(fig,OUT/"04_prediction_evidence",caption)


def load_modern_missingness(experiment,work):
    """Read only complete, matched-mask OOF results; verify their plotted values."""
    if experiment is None:return None
    required=[experiment/name for name in ["status.json","comparison.csv","oof_predictions.npz",
                                           "paired_ci_vs_knn.csv","experiment_spec.json"]]
    if not all(path.is_file() for path in required):return None
    state=json.loads((experiment/"status.json").read_text())
    if not state.get("complete",False) or state.get("completed_folds")!=5:return None
    read_json(experiment/"status.json");read_json(experiment/"experiment_spec.json")
    metrics=read_csv(experiment/"comparison.csv")
    intervals=read_csv(experiment/"paired_ci_vs_knn.csv")
    arrays=read_npz(experiment/"oof_predictions.npz")
    original=read_npz(work/"q1/oof_arrays.npz")
    folds=read_csv(work/"q1/fold_assignments.csv").set_index("beer")
    np.testing.assert_array_equal(arrays["Y"],original["Y"])
    np.testing.assert_array_equal(arrays["fold"],folds.loc[arrays["beer"],"fold"])
    y=arrays["Y"];fold=arrays["fold"]
    assert y.shape==(250,50) and all((fold==f).sum()==50 for f in range(5))
    core=np.array([str(t).startswith(("A_","F_")) for t in arrays["targets"]])
    assert core.sum()==30
    cut=np.stack([np.quantile(y[fold!=f],np.arange(1,7)/7,axis=0) for f in range(5)])
    def grades(pred):
        z=np.empty(pred.shape,dtype=int)
        for f in range(5):
            rows=fold==f
            for j in range(50):z[rows,j]=np.sum(pred[rows,j,None]>=cut[f,:,j],axis=1)
        return z
    truth=grades(y);den=((y-y.mean(0))**2).sum(0)
    model_names=["old_median","old_knn","tabicl_raw231_y16"]
    scenarios=["original","derived_only_missing","MCAR_10_strict","MCAR_30_strict","MCAR_50_strict"]
    for model in model_names:
        for scenario in scenarios:
            pred=arrays[f"{model}__{scenario}"]
            assert pred.shape==y.shape and np.isfinite(pred).all()
            r2=1-((y-pred)**2).sum(0)/den;error=abs(grades(pred)-truth)
            for group,mask in [("all50",np.ones(50,bool)),("core30",core)]:
                row=metrics[(metrics.model==model)&(metrics.scenario==scenario)&(metrics.targets==group)]
                assert len(row)==1
                observed=[r2[mask].mean(),error[:,mask].mean(),(error[:,mask]==0).mean(),(error[:,mask]<=1).mean()]
                reported=row[["r2","grade_mae","grade_exact","grade_within_one"]].to_numpy(float)[0]
                np.testing.assert_allclose(observed,reported,atol=1e-12,rtol=0)
    export_rows("05_modern_missingness_all",metrics)
    export_rows("05_modern_paired_ci_vs_knn",intervals)
    return metrics


def missingness(work,modern_dir=None):
    d=read_csv(work/"competition_revision/grading/metrics.csv")
    z=d[d.target_group=="核心30维"]
    scenarios=["derived_only_missing","MCAR_10_strict","MCAR_30_strict","MCAR_50_strict"]
    xs=np.array([0,10,30,50])
    modern=load_modern_missingness(modern_dir,work)
    if modern is not None:
        now=modern[modern.targets=="core30"]
        for old_name,new_name in [("中位数插补","old_median"),("KNN插补","old_knn")]:
            original=z[z.method==old_name].set_index("scenario").loc[scenarios]
            rerun=now[now.model==new_name].set_index("scenario").loc[scenarios]
            np.testing.assert_allclose(original.macro_r2,rerun.r2,atol=1e-12,rtol=0)
            np.testing.assert_allclose(original.within_one_fraction,rerun.grade_within_one,atol=1e-12,rtol=0)
    fig,axs=plt.subplots(1,2,figsize=(8.6,4.0))
    fig.subplots_adjust(left=.095,right=.96,bottom=.23,top=.79,wspace=.34)
    all_r2=[];all_grade=[]
    for method,color,marker in [("中位数插补",C["gray"],"s"),("KNN插补",C["teal"],"o")]:
        a=z[z.method==method].set_index("scenario").loc[scenarios]
        all_r2.extend(a.macro_r2);all_grade.extend(100*a.within_one_fraction)
        for ax,vals in zip(axs,[a.macro_r2,a.within_one_fraction*100]):
            label=method if modern is None else "六模型＋"+method.removesuffix("插补")
            ax.plot(xs,vals,marker=marker,color=color,lw=1.8,label=label)
        for ax,vals,fmt in [(axs[0],a.macro_r2,".3f"),(axs[1],100*a.within_one_fraction,".1f")]:
            ax.annotate(format(vals.iloc[-1],fmt),(50,vals.iloc[-1]),xytext=(6,0),textcoords="offset points",va="center",fontsize=10,color=color)
    if modern is not None:
        a=now[now.model=="tabicl_raw231_y16"].set_index("scenario").loc[scenarios]
        all_r2.extend(a.r2);all_grade.extend(100*a.grade_within_one)
        for ax,values in zip(axs,[a.r2,100*a.grade_within_one]):
            ax.plot(xs,values,marker="D",color=C["orange"],lw=1.8,label="TabICL · Y16",zorder=4)
        for ax,values,fmt in [(axs[0],a.r2,".3f"),(axs[1],100*a.grade_within_one,".1f")]:
            ax.annotate(format(values.iloc[-1],fmt),(50,values.iloc[-1]),xytext=(6,0),textcoords="offset points",va="center",fontsize=10,color=C["orange"])
    for ax in axs:
        ax.set(xlim=(-2,56),xticks=xs,xlabel="其余 226 字段的随机缺失率 / %")
        clean(ax,"y")
    axs[0].set(ylabel="核心 30 维 · R²",ylim=(min(.10,min(all_r2)-.015),max(.245,max(all_r2)+.015)))
    axs[1].set(ylabel="相对等级误差 ≤ 1 档 / %",ylim=(min(47,min(all_grade)-1),max(58,max(all_grade)+1)))
    panel_title(axs[0],"a","连续感官值的缺失压力")
    panel_title(axs[1],"b","相对等级的可用性")
    if modern is None:axs[0].legend(loc="lower left",bbox_to_anchor=(-.01,1.12),ncol=2)
    else:fig.legend(*axs[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.55,.99),ncol=3,columnspacing=1.6)
    fig.text(.095,.07,"250 款啤酒 · 固定五折；5 项派生字段始终屏蔽，其余 226 项含技术内标。",fontsize=9.3,color=C["text"])
    export_rows("05_missingness",z[z.scenario.isin(scenarios)])
    caption="严格缺失压力测试。两种插补均使用原五折划分与固定预测规格，验证输入的派生字段始终屏蔽，再增加随机测量缺失。相对 0–6 等级依据各训练折的七分位点构造，非原始品评档位。线仅连接实测缺失场景，不表示场景之间有额外实验。"
    if modern is not None:caption="相同五折和相同缺测掩码下的预测压力测试。六模型集成分别采用中位数与KNN插补；TabICL使用已冻结的原始231列至16维目标表示。各方法训练于原训练输入，测试时对231列施加相同的10%、30%、50%随机掩码，并始终屏蔽5项派生字段；其余226列包含技术内标。0%表示仅屏蔽派生字段，完整原输入结果另保留在绘图数据表。相对0–6等级由各训练折七分位点构造。线仅连接实际执行的场景；配对差值区间单独保存，不画作绝对性能区间。"
    save(fig,OUT/"05_missingness_evidence",caption)


def force_layout(nodes,edges,seed=20261024,iterations=500):
    """Deterministic force-directed layout; positions carry no measurement units."""
    n=len(nodes);idx={t:i for i,t in enumerate(nodes)}
    rng=np.random.default_rng(seed);p=rng.normal(0,.25,(n,2))
    A=np.zeros((n,n))
    for r in edges.itertuples():
        i,j=idx[r.source],idx[r.target];A[i,j]=A[j,i]=abs(r.rho)
    k=1/np.sqrt(n)
    for it in range(iterations):
        delta=p[:,None,:]-p[None,:,:]
        dist=np.maximum(np.linalg.norm(delta,axis=-1),.015)
        force=k*k/(dist*dist)-A*dist/k
        np.fill_diagonal(force,0)
        dp=(delta*force[:,:,None]).sum(axis=1)-p*.09
        length=np.maximum(np.linalg.norm(dp,axis=1),1e-9)
        temp=.06*(1-it/iterations)+.0005
        p+=dp/length[:,None]*np.minimum(length,temp)[:,None]
        p-=p.mean(axis=0)
    p/=np.abs(p).max()
    return dict(zip(nodes,p))


def network_layout(connected,edges):
    """Pack connected components; enlarge the main component without hiding nodes."""
    neighbors={t:set() for t in connected.field}
    for r in edges.itertuples():
        neighbors[r.source].add(r.target);neighbors[r.target].add(r.source)
    remaining=set(neighbors);components=[]
    while remaining:
        start=min(remaining);stack=[start];component=set()
        while stack:
            u=stack.pop()
            if u not in component:
                component.add(u);stack.extend(sorted(neighbors[u]-component))
        components.append(sorted(component));remaining-=component
    components.sort(key=lambda c:(-len(c),c[0]))
    main=components[0]
    pos=force_layout(main,edges[edges.source.isin(main)&edges.target.isin(main)])
    pos={n:np.array([p[0]*.86,p[1]*.86+.18]) for n,p in pos.items()}
    others=components[1:]
    for i,comp in enumerate(others):
        cx=-.95+(i%10)*.21;cy=-.83-(i//10)*.15
        for j,n in enumerate(comp):
            a=2*np.pi*j/len(comp);r=.032 if len(comp)<=3 else .049
            pos[n]=np.array([cx+r*np.cos(a),cy+r*np.sin(a)])
    return pos,components


def network_modules(work,names):
    nodes=read_csv(work/"q2/graph_nodes.csv")
    edges=read_csv(work/"q2/graph_edges.csv")
    lead=read_csv(work/"competition_revision/module_stability/leading_module_by_target.csv")
    summary=read_json(work/"competition_revision/module_stability/summary.json")
    z=lead[(lead.threshold==.6)&(~lead.any_individual_at_least_0_8)&(lead.module_union_frequency>=.8)].sort_values("module_union_frequency",ascending=False)
    fig=plt.figure(figsize=(8.6,4.8))
    # A tall graph and two stacked evidence panels retain readable A4 labels.
    ax=fig.add_axes([.025,.18,.43,.70])
    connected=nodes[nodes.degree>0].copy();pos,components=network_layout(connected,edges)
    p=np.array([pos[t] for t in connected.field])
    positive=[(pos[r.source],pos[r.target]) for r in edges.itertuples() if r.rho>=0]
    negative=[(pos[r.source],pos[r.target]) for r in edges.itertuples() if r.rho<0]
    ax.add_collection(LineCollection(positive,colors=C["gray"],linewidths=.55,alpha=.62,zorder=1))
    ax.add_collection(LineCollection(negative,colors=C["reject"],linewidths=.85,linestyles="dashed",alpha=.9,zorder=2))
    palette={1:C["teal"],2:C["orange"],3:C["slate"],4:C["olive"],5:C["reject"]}
    ax.scatter(p[:,0],p[:,1],s=10+connected.weighted_degree.to_numpy()*2.1,
               c=[palette.get(m,C["gray"]) for m in connected.module],edgecolors="white",linewidths=.35,zorder=3)
    ax.set(xlim=(-1.08,1.08),ylim=(-1.59,1.10));ax.set_aspect("equal");ax.axis("off")
    # Actual isolated nodes are shown separately, retaining all 231 nodes.
    iso=nodes[nodes.degree==0]
    for j in range(len(iso)):
        ax.plot(-.91+(j%34)*.055,-1.28-(j//34)*.057,"o",color=C["gray"],markersize=1.8)
    ax.text(0,-1.53,f"{len(iso)} 个孤立节点",ha="center",fontsize=10,color=C["text"])
    panel_title(ax,"a",f"{len(nodes)} 节点 · {len(edges)} 条关联",y=1.04)
    ax.text(.01,1.005,"|ρ| ≥ 0.60 · 布局无物理尺度",transform=ax.transAxes,fontsize=10,color=C["text"])
    # The layout is exported with identities to make the network reproducible.
    pp=connected[["field","module","degree","weighted_degree"]].copy();pp["layout_x"]=p[:,0];pp["layout_y"]=p[:,1]
    export_rows("06_network_positions",pp)
    export_rows("06_network_edges",edges)
    ax=fig.add_axes([.65,.58,.32,.28]);yy=np.arange(len(z))
    ax.hlines(yy,z.largest_member_frequency,z.module_union_frequency,color=C["line"],lw=2.6)
    ax.scatter(z.largest_member_frequency,yy,c=C["orange"],s=34,zorder=3)
    ax.scatter(z.module_union_frequency,yy,c=C["teal"],s=34,zorder=3)
    ax.scatter(z.uniform_three_draw_reference,yy,marker="x",c=C["gray"],s=25,zorder=2,linewidths=1.2)
    ax.axvline(.8,color=C["gray"],ls=(0,(3,3)),lw=.8)
    ax.set(yticks=yy,yticklabels=[f"{names[t]} ({n})" for t,n in zip(z.target,z.eligible_module_size)],
           xlim=(0,1.02),ylim=(len(z)-.55,-.65),xlabel="400 次 bootstrap 入选频率")
    ax.xaxis.set_major_locator(FixedLocator([0,.2,.4,.6,.8,1]))
    clean(ax,"x");panel_title(ax,"b","成员轮换 · 模块稳定",y=1.10)
    ax.legend(handles=[dot_handle(C["teal"],"模块并集"),dot_handle(C["orange"],"最稳定成员"),dot_handle(C["gray"],"大小参照",marker="x")],
              ncol=3,fontsize=9.8,handletextpad=.12,columnspacing=.5,loc="lower center",bbox_to_anchor=(.40,1.22))
    ax=fig.add_axes([.65,.15,.32,.26]);ss=pd.DataFrame(summary["threshold_summary"])
    xs=ss.threshold.to_numpy()
    ax.plot(xs,ss.at_least_one_module_frequency_0_8,"o-",color=C["teal"],label="模块")
    ax.plot(xs,ss.at_least_one_individual_frequency_0_8,"s--",color=C["orange"],label="单化合物")
    for x,y in zip(xs,ss.at_least_one_module_frequency_0_8):ax.annotate(str(y),(x,y),xytext=(0,7),textcoords="offset points",ha="center",fontsize=10,color=C["teal"])
    ax.set(xlim=(.475,.825),xticks=xs,ylim=(0,31),yticks=[0,10,20,30],xlabel="共变阈值 |ρ|",ylabel="稳定感官数 / 50")
    clean(ax,"y");panel_title(ax,"c","阈值敏感性",y=1.10)
    ax.legend(loc="lower left",fontsize=10)
    fig.text(.04,.064,"b：全部 6 个“成员 < 80%、模块 ≥ 80%”目标；括号为模块大小。",fontsize=10,color=C["text"])
    fig.text(.04,.025,"c：“稳定”指至少一个成员或模块的入选频率达到 80%。",fontsize=10,color=C["text"])
    export_rows("06_module_substitution",z);export_rows("06_threshold_sensitivity",ss)
    save(fig,OUT/"06_network_module_evidence","观测化学的描述性共变图及模块稳定性。网络包含全部 231 节点；130 个有连接节点按确定性力导向布局并按连通分量分区排列，101 个孤立节点独立展示。节点色表示前五个模块及其他模块，虚线标记 3 条负相关边。图 b 的模块频率是真实 bootstrap Top-3 集合的并集事件，并非单成员频率之和；灰色叉为均匀抽取的模块大小参照，不是 p 值。模块固定于全数据描述性图，结果不代表图划分本身的重采样稳定性。")


def generation(work):
    folds=read_csv(work/"q3/extension_outer_comparison.csv")
    intervals=read_csv(work/"q3/extension_paired_energy_intervals.csv")
    selected=read_json(work/"q3/extension_selection_process.json")
    order=["smooth_pooled","interval_gaussian","interval_kernel_h03","interval_kernel_h06","interval_kernel_h09"]
    labels=["平滑 Copula","区间 Gaussian","区间核 h=0.3","区间核 h=0.6","区间核 h=0.9"]
    fig=plt.figure(figsize=(8.6,6.6))
    first=fig.add_axes([.21,.61,.32,.24])
    second=fig.add_axes([.64,.61,.33,.24],sharey=first)
    third=fig.add_axes([.21,.20,.76,.25])
    axs=[first,second,third]
    means=folds.groupby("model")[["c2st_auc","nn_ratio","energy"]].mean().loc[order]
    offsets={0:-.12,1:0,2:.12}
    for ax,key in zip(axs[:2],["c2st_auc","nn_ratio"]):
        for i,m in enumerate(order):
            z=folds[folds.model==m]
            yy=np.array([i+offsets[int(f)] for f in z.outer_fold])
            ax.scatter(z[key],yy,s=20,c=C["gray"],marker="o",alpha=.8,zorder=2)
            ax.scatter([means.loc[m,key]],[i],s=46,c=C["teal"] if m=="interval_gaussian" else C["orange"],marker="D",edgecolors="white",linewidths=.6,zorder=4)
        clean(ax,"x")
    axs[0].axvline(.5,color=C["text"],ls=(0,(3,3)),lw=.9)
    axs[0].set(xlim=(.48,1),xticks=[.5,.65,.8,.95],yticks=np.arange(5),yticklabels=labels,ylim=(4.6,-.6),xlabel="C2ST AUC · 0.5 为随机猜测")
    axs[1].tick_params(labelleft=False)
    axs[1].axvline(1,color=C["text"],ls=(0,(3,3)),lw=.9)
    axs[1].set(xlim=(.2,1.2),xticks=[.25,.5,.75,1],xlabel="生成 / 真实的近邻距离比")
    z=intervals.set_index("reference").loc[order]
    for i,m in enumerate(order):
        r=z.loc[m];v=r.selected_process_energy_reduction
        axs[2].errorbar(v,i,xerr=[[v-r.ci95_low],[r.ci95_high-v]],fmt="o",ms=5,color=C["teal"],lw=1.7,capsize=3)
    axs[2].axvline(0,color=C["text"],lw=.9)
    axs[2].set(xlim=(-.002,.02),xticks=[0,.005,.01,.015,.02],
               yticks=np.arange(5),yticklabels=labels,ylim=(4.6,-.6),xlabel="参照能量分数 − 选择过程能量分数")
    clean(axs[2],"x")
    panel_title(axs[0],"a","真假可分性",y=1.12)
    panel_title(axs[1],"b","邻近训练集的程度",y=1.12)
    panel_title(axs[2],"c","配对能量改善与 95% 区间",y=1.13)
    fig.text(.06,.085,"灰点：3 个外层折；菱形：等折均值。c：250 款的配对 bootstrap 区间。",fontsize=10,color=C["text"])
    fig.text(.06,.039,"综合分布差异、真假可分性与近邻程度采用独立指标衡量。",fontsize=10,color=C["text"])
    export_rows("07_generation_outer_folds",folds);export_rows("07_generation_paired_intervals",intervals)
    export_rows("07_generation_means",means.reset_index())
    save(fig,OUT/"07_generation_quality","生成质量的三重证据。上方 a、b 的灰点是三折内部嵌套验证的实际折级结果，菱形是等折均值，并非置信区间；真假分类 AUC 越接近 0.5 越难区分，近邻距离比接近 1 表示与真实样本到训练集的距离相当。下方 c 采用已保存的 250 款配对重采样区间，正值表示嵌套选择过程的能量分数更低。选择过程相对区间 Gaussian 的区间跨零，不能据此声称稳定优越。该研究是诊断后内部验证，不是新的外部锁定测试。")


def semantic(work,experiment):
    branch=read_csv(experiment/"model_comparison.csv")
    blend=read_csv(experiment/"blend_comparison.csv")
    contrasts=read_json(experiment/"semantic_contrasts.json")
    protocol=read_json(experiment/"protocol.json")
    old=read_csv(work/"q1/dimension_metrics.csv")
    old=old[old.model=="nested_ensemble"]
    baseline=np.array([old.r2.mean(),old.loc[old.target.str.startswith(("A_","F_")),"r2"].mean()])
    assert protocol["n"]==250 and protocol["outer_folds"]==5 and protocol["inner_folds"]==3
    new_folds=read_csv(experiment/"fold_assignments.csv")
    old_folds=read_csv(work/"q1/fold_assignments.csv")
    np.testing.assert_array_equal(new_folds.fold,old_folds.fold)
    conditions=["no_aux","aligned_aux","permuted_aux"]
    labels=["无辅助","真实语义","置乱语义"]
    a=branch.set_index("model").loc[[c+"_all50" for c in conditions]]
    b=blend.set_index("model").loc[["nested_blend_"+c+"_all50" for c in conditions]]
    fig,axs=plt.subplots(1,2,figsize=(8.6,4.4),sharey=True,
                         gridspec_kw={"width_ratios":[1,1.16]})
    fig.subplots_adjust(left=.13,right=.97,bottom=.24,top=.76,wspace=.28)
    yy=np.arange(3)
    for i,row in enumerate(a.itertuples()):
        axs[0].plot([row.core30_r2,row.all50_r2],[i,i],color=C["line"],lw=2.2,zorder=1)
    for j,(key,color,marker,filled) in enumerate([("all50_r2",C["teal"],"o",True),("core30_r2",C["orange"],"o",False)]):
        axs[0].scatter(a[key],yy,s=47,c=color if filled else "white",edgecolors=color,linewidths=1.4,zorder=3)
        for i,v in enumerate(a[key]):
            axs[0].annotate(f"{v:.4f}",(v,i),xytext=(0,10 if filled else -18),textcoords="offset points",ha="center",fontsize=10,color=color)
        delta=b[key].to_numpy()-baseline[j]
        intervals=np.array([json.loads(t) for t in b["ci_vs_frozen_all50" if j==0 else "ci_vs_frozen_core30"]])
        offset=-.14 if j==0 else .14
        axs[1].errorbar(delta*1000,yy+offset,xerr=np.vstack([delta-intervals[:,0],intervals[:,1]-delta])*1000,
                        fmt="o",ms=6,mfc=color if filled else "white",mec=color,mew=1.3,color=color,lw=1.6,capsize=3,zorder=3)
    axs[0].set(xlim=(.192,.260),xticks=[.20,.22,.24,.26],yticks=yy,yticklabels=labels,ylim=(2.6,-.6),xlabel="单独 RRR 分支 · 宏平均 R²")
    axs[1].axvline(0,color=C["text"],lw=1,zorder=1)
    axs[1].set(xlim=(-.8,3.9),xticks=[0,1,2,3],xlabel="相对冻结六模型 · ΔR²（单位：0.001）")
    panel_title(axs[0],"a","辅助信息的三条件对照")
    panel_title(axs[1],"b","融合后的改善与 95% 区间")
    for ax in axs:clean(ax,"x")
    fig.legend(handles=[dot_handle(C["teal"],"全部 50 维"),dot_handle(C["orange"],"香气 / 风味 30 维",filled=False)],
               loc="upper center",bbox_to_anchor=(.55,.98),ncol=2,columnspacing=2)
    fig.text(.13,.09,"全部规格与融合权重在训练折内选择；右图为固定折外预测的啤酒级配对 bootstrap（2000 次）。",fontsize=9.2,color=C["text"])
    fig.text(.13,.035,"两图均采用 50 维内层损失选择规格；核心 30 维保持为同一预定目标子集。",fontsize=9.2,color=C["text"])
    export_rows("09_semantic_all_conditions",pd.concat([branch.assign(stage="RRR"),blend.assign(stage="nested_blend")],ignore_index=True))
    export_rows("09_semantic_direct_contrasts",pd.DataFrame(contrasts))
    save(fig,OUT/"09_semantic_auxiliary_evidence","训练期语义辅助的对照实验。所有模型在同一五折外层划分、三折内层选择下评价；真实语义与化学样本对齐，置乱语义仅在每个训练分区内随机换序，无辅助分支设语义权重为零。推断仅使用化学输入。左图报告 RRR 单分支结果，右图报告与冻结六模型融合后的配对 R² 改善及 2000 次啤酒级 bootstrap 的 95% 区间。真实语义融合的 50 维改善为 0.0019605，区间 [0.0004937, 0.0032640]；核心 30 维改善为 0.0017095，区间 [-0.0001207, 0.0034217]。增益较小，核心任务区间跨零。按核心 30 维选择规格的完整对照另保留于绘图数据表。")


def semantic_style_controls(experiment):
    """Each permutation remains a separate matched control, never an ensemble."""
    path=experiment/"style_controls"
    if not (path/"COMPLETE.json").is_file():return
    complete=read_json(path/"COMPLETE.json")
    if not complete.get("complete"):return
    assert complete["fitted_outer_models"]==60 and complete["permutation_replicates"]==5
    protocol=read_json(path/"protocol.json")
    assert protocol["controls_not_ensembled"] and protocol["bootstrap_replicates"]==2000
    contrasts=read_json(path/"contrasts.json")
    metrics=read_csv(path/"model_comparison.csv")
    selected=[r for r in contrasts if r["stage"]=="blend" and r["control"].endswith("_all50")]
    perms=sorted([r for r in selected if r["control"].startswith("within_style_")],key=lambda r:r["control"])
    style=next(r for r in selected if r["control"]=="style_only_all50")
    assert len(perms)==5
    rows=perms+[style]
    yy=np.r_[np.arange(5),5.65]
    labels=[f"风格内置乱 {i}" for i in range(1,6)]+["仅风格标签†"]
    fig,axs=plt.subplots(1,2,figsize=(8.6,4.5),sharey=True,sharex=True)
    fig.subplots_adjust(left=.18,right=.97,bottom=.27,top=.86,wspace=.18)
    chart_rows=[]
    for ax,group,color,letter,title in zip(axs,["all50","core30"],[C["teal"],C["orange"]],
                                          ["a","b"],["全部 50 维","香气 / 风味 30 维"]):
        ax.axvline(0,color=C["ink"],lw=1,zorder=1)
        ax.axhline(4.85,color=C["line"],lw=.8,zorder=1)
        for i,row in enumerate(rows):
            point=1000*row[group+"_delta"];low,high=np.array(row[group+"_ci"])*1000
            ink=C["slate"] if i==5 else color
            ax.errorbar(point,yy[i],xerr=[[point-low],[high-point]],fmt="D" if i==5 else "o",
                        color=ink,mfc="white" if i==5 else ink,mec=ink,mew=1.2,
                        ms=5.6,capsize=3,lw=1.6,zorder=3)
            chart_rows.append(dict(control=row["control"],selection="all50",targets=group,
                                   delta_r2=row[group+"_delta"],ci025=row[group+"_ci"][0],ci975=row[group+"_ci"][1],
                                   matched_auxiliary_dimension=i<5,control_ensembled=False))
        ax.set(xlim=(-1.9,4.05),xticks=[-1,0,1,2,3,4],yticks=yy,yticklabels=labels,
               ylim=(6.3,-.65),xlabel="真实语义 − 各控制 · ΔR²（单位：0.001）")
        clean(ax,"x");panel_title(ax,letter,title,y=1.055)
    fig.text(.18,.135,"两图均以全部 50 维的内层损失选参；各置乱在每个实际训练分区内独立构造。",fontsize=9.6,color=C["text"])
    fig.text(.18,.080,"线段：2000 次固定折外预测的配对 95% 区间，未经多重比较校正；五次控制分别报告。",fontsize=9.3,color=C["text"])
    fig.text(.18,.025,"† 仅风格标签是辅助任务替换诊断，其辅助空间及有效秩与高维词频任务不同。",fontsize=9.3,color=C["text"])
    export_rows("13_semantic_style_selected",pd.DataFrame(chart_rows))
    export_rows("13_semantic_style_all_models",metrics)
    flat=[]
    for row in contrasts:
        for group in ["all50","core30"]:
            flat.append(dict(stage=row["stage"],control=row["control"],targets=group,
                             delta_r2=row[group+"_delta"],ci025=row[group+"_ci"][0],ci975=row[group+"_ci"][1]))
    export_rows("13_semantic_style_all_contrasts",pd.DataFrame(flat))
    save(fig,OUT/"13_semantic_style_controls","真实评论配对信息的风格条件控制。全部模型采用同一外五折、内三折，图中均按 50 维内层损失选参并在训练折内选择融合权重。每个点表示真实对齐辅助融合相对该行单个控制的宏平均 R² 差；线段为 2000 次固定折外预测的配对啤酒 bootstrap 95% 百分位区间。五个风格内置乱控制保持评论维数、风格归属与候选规格搜索空间，分别在每个实际训练分区构造；不将五个控制预测平均成新的集成。真实配对的 50 维增量均为小幅正值，核心 30 维有区间跨零；各区间未经多重比较校正。分隔线下的风格标签辅助属于任务替换诊断，辅助空间和有效秩不同，真实评论相对它的区间跨零。两种选参目标、两种阶段的完整数值均保留在绘图数据表。")


def classification_certificate(path):
    d=read_csv(path).sort_values("unavoidable_distance_increase_lower_bound",ascending=False).reset_index(drop=True)
    assert d.all_continuous_points_fail_positive_improvement.all()
    assert d.ellipsoid_single_halfspace_exclusion.all()
    assert (d.linear_mip_gap==0).all()
    assert np.max(np.abs(d.primal_dual_gap))<1e-10
    fig,axs=plt.subplots(1,2,figsize=(8.6,5.3),sharey=True,gridspec_kw={"width_ratios":[1.15,1]})
    fig.subplots_adjust(left=.14,right=.97,bottom=.26,top=.83,wspace=.32)
    yy=np.arange(len(d));increase=100*d.unavoidable_distance_increase_lower_bound
    axs[0].hlines(yy,0,increase,color=C["line"],lw=2)
    axs[0].scatter(increase,yy,c=C["reject"],s=43,marker=">",zorder=3)
    for y,v in zip(yy,increase):axs[0].annotate(f"{v:.1f}%",(v,y),xytext=(7,0),textcoords="offset points",va="center",fontsize=9.4,color=C["reject"])
    axs[0].set(xlim=(0,103),xticks=[0,20,40,60,80,100],yticks=yy,
               yticklabels=[f"{b}  ·  k={s}" for b,s in zip(d.beer,d.support_size)],
               ylim=(len(d)-.55,-.7),xlabel="目标距离增幅的数值下界 / %")
    bound=d.ellipsoid_single_halfspace_min_upper_margin
    axs[1].hlines(yy,bound,0,color=C["line"],lw=2)
    axs[1].scatter(bound,yy,c=C["teal"],s=35,zorder=3)
    for y,v in zip(yy,bound):axs[1].annotate(f"{v:.2f}",(v,y),xytext=(-6,0),textcoords="offset points",ha="right",va="center",fontsize=9.4,color=C["teal"])
    axs[1].axvline(0,color=C["text"],lw=1)
    axs[1].set(xlim=(-6.8,.35),xticks=[-6,-4,-2,0],xlabel="单半空间排除界 · logit 裕量")
    for ax in axs:clean(ax,"x")
    panel_title(axs[0],"a","最少分类编辑的几何代价")
    panel_title(axs[1],"b","距离椭球内的解析排除")
    fig.text(.14,.11,"每行对应一个返回的固定支持；标签为源 beer 编号与支持大小。",fontsize=9.3,color=C["text"])
    fig.text(.14,.055,"左：凸二次问题的对偶下界。右：椭球支持函数给出的分类裕量上界；全部低于可达阈值。",fontsize=9.3,color=C["text"])
    export_rows("10_classification_geometry_certificates",d)
    save(fig,OUT/"10_classification_geometry_certificate","分类可达不蕴含几何靠近的数值证书。对 11 款源啤酒，分类 MILP 返回 5–10 个字段的最少编辑支持；在每个返回的固定支持上，再对所有连续候选求解正定凸二次距离问题。原对偶间隙的绝对值均小于 7.4×10⁻¹³，对偶下界表明目标距离至少增加 18.4%–83.5%。右图在源距离椭球上计算各分类半空间线性函数的支持值，取最小可达上界；负值给出一个半空间已不可满足的独立排除证据。该结论覆盖这些固定支持上的全部连续候选，未穷举同等大小的其他支持；全坐标联合 MIQCP 负责后者的搜索与下界。")


def within_style(experiment):
    associations=read_csv(experiment/"full_data_pair_evidence.csv").set_index(["target","field"])
    stability=read_csv(experiment/"fixed_alpha_pair_evidence.csv").set_index(["target","field"])
    prediction=read_csv(experiment/"oof_paired_differences.csv")
    prediction=prediction[(prediction.model=="within_linear")&(prediction.baseline=="style_mean_raw")].set_index("target")
    protocol=read_json(experiment/"protocol.json")
    fixed=read_json(experiment/"fixed_alpha_summary.json")
    assert protocol["n_samples"]==250 and protocol["candidate_compounds"]==218
    assert fixed["alpha"]==3 and fixed["replicates"]==400
    keys=[
        ("lactic","lactic_acid.mg.L.","乳酸 → 乳酸感"),
        ("acidity","lactic_acid.mg.L.","乳酸 → 酸味"),
        ("X4vg","X4VG","4-VG → 酚香"),
        ("alcohol","ethanol..v.v.","乙醇 → 酒精感"),
        ("A_malt_all","X2.acetylfuran","2-乙酰呋喃 → 麦芽香气"),
        ("F_malt_all","X2.6.dimethyl.pyrazine","2,6-DMP → 麦芽风味"),
        ("dms","DMS","DMS → DMS 感"),
        ("diacetyl","X2.3.butanedione","双乙酰 → 双乙酰感"),
    ]
    rows=[]
    for target,field,label in keys:
        a=associations.loc[(target,field)];b=stability.loc[(target,field)]
        rows.append(dict(target=target,field=field,label=label,
                         correlation_total=a.correlation_total,correlation_within=a.correlation_within,
                         total_top3=b.total_linear_fixed_top3_frequency,
                         within_top3=b.within_linear_fixed_top3_frequency,
                         covariance_total=a.covariance_total,covariance_within=a.covariance_within,
                         covariance_between=a.covariance_between))
    d=pd.DataFrame(rows);yy=np.arange(len(d))
    fig=plt.figure(figsize=(8.6,5.9))
    top=fig.add_gridspec(1,2,left=.255,right=.97,bottom=.55,top=.885,wspace=.30)
    ax=fig.add_subplot(top[0]);bx=fig.add_subplot(top[1],sharey=ax)
    for panel,left,right in [(ax,d.correlation_total,d.correlation_within),(bx,d.total_top3,d.within_top3)]:
        panel.hlines(yy,left,right,color=C["line"],lw=2.3)
        # The hollow total marker remains visible if both endpoints coincide.
        panel.scatter(left,yy,s=49,facecolors="white",edgecolors=C["orange"],linewidths=1.5,zorder=3)
        panel.scatter(right,yy,s=24,c=C["teal"],zorder=4)
        panel.set(ylim=(len(d)-.4,-.6),yticks=yy)
        panel.tick_params(axis="y",length=0,labelsize=10.5)
        clean(panel,"x")
    ax.set(yticklabels=d.label,xlim=(-.20,.95),xticks=[0,.25,.5,.75],xlabel="全样本正态分数相关")
    ax.axvline(0,color=C["gray"],lw=.8)
    bx.tick_params(labelleft=False)
    bx.set(xlim=(-.07,1.07),xticks=[0,.25,.5,.75,1],xlabel="Top 3 入选频率 · 原感官尺度")
    bx.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{int(round(100*x))}%"))
    panel_title(ax,"a","总体与风格内关联",y=1.075)
    panel_title(bx,"b","同惩罚下的排名稳定性",y=1.075)
    fig.legend(handles=[dot_handle(C["orange"],"总体",filled=False),dot_handle(C["teal"],"风格内")],
               loc="upper center",bbox_to_anchor=(.60,.995),ncol=2,columnspacing=2.3)
    cx=fig.add_axes([.095,.17,.875,.215])
    targets=[("__macro__","全部\n50 维"),("acidity","酸味"),("alcohol","酒精感"),
             ("lactic","乳酸感"),("A_malt_all","麦芽\n香气"),("F_malt_all","麦芽\n风味"),
             ("X4vg","酚香"),("diacetyl","双乙酰感"),("dms","DMS 感"),("t2n","t2n 感")]
    pred_rows=prediction.loc[[k for k,_ in targets]].copy()
    for i,row in enumerate(pred_rows.itertuples()):
        color=C["ink"] if i==0 else C["teal"] if row.paired_bootstrap_025>0 else C["gray"]
        cx.errorbar(i,row.delta_R2,
            yerr=[[row.delta_R2-row.paired_bootstrap_025],[row.paired_bootstrap_975-row.delta_R2]],
            fmt="D" if i==0 else "o",ms=6 if i==0 else 5,color=color,lw=1.6,capsize=3,zorder=3)
    cx.axhline(0,color=C["text"],lw=.9)
    cx.set(xlim=(-.55,len(targets)-.45),xticks=np.arange(len(targets)),xticklabels=[t for _,t in targets],
           ylim=(-.052,.118),yticks=[-.04,0,.04,.08],ylabel="相对风格均值的 ΔR²")
    cx.tick_params(axis="x",labelsize=10.3)
    clean(cx,"y")
    panel_title(cx,"c","风格内全化学模型的折外增量",y=1.10)
    macro=prediction.loc["__macro__"]
    cx.text(1,1.12,f"50 维：+{macro.delta_R2:.4f}  [{macro.paired_bootstrap_025:.4f}, {macro.paired_bootstrap_975:.4f}]",
            transform=cx.transAxes,ha="right",va="bottom",fontsize=9.4,color=C["teal"])
    fig.text(.06,.055,"a：完整样本的描述性关联。b：固定 α=3，400 次风格分层重拟合。",fontsize=9.5,color=C["text"])
    fig.text(.06,.024,"c：2000 次风格内配对损失重采样；固定折模型及原始 Y 全样本方差。",fontsize=9.5,color=C["text"])
    export_rows("11_within_style_chemical_evidence",d)
    export_rows("11_within_style_prediction_increment",pred_rows.reset_index())
    save(fig,OUT/"11_within_style_evidence","风格内检验重新排列化学候选。a 对照全样本正态分数尺度的总体相关与去除各风格均值后的相关；b 使用相同 400 组风格分层重采样索引、相同 α=3 和原始感官尺度，对照总体/风格内 Top-3 稳定性，空心与实心标记保留重合端点。2,6-DMP 指 2,6-二甲基吡嗪。4-VG 的对应证据稳定，乳酸两项总体高排名在风格内消失；原量表下乳酸仍可保留较弱正关联，排名下降不等于作用消失。c 是使用全部化学字段的风格内模型相对于仅风格均值的折外预测增量，不是某一化合物的效应。全部 50 维宏平均增量为 0.041812，95% 区间 [0.032215,0.052464]；区间来自固定折模型、固定全样本目标方差下的风格内配对损失重采样，未包含重训模型的不确定性。")


def transfer(work,extra=None):
    z=read_csv(work/"competition_revision/style_feasibility/joint_budget_summary.csv")
    rows=[]
    for r in z.itertuples():
        for method,label,value,n in [
            ("distance","仅距离目标",r.old_distance_objective_mean_distance_reduction,r.old_distance_objective_tripel),
            ("class_only","仅分类筛选",r.class_only_all_11_mean_distance_reduction,r.class_only_accepted),
            ("joint","原支持池联合筛选",r.all_11_mean_distance_reduction,r.joint_accepted)]:
            rows.append(dict(method=method,label=label,budget=r.budget,n_sources=r.n_sources,
                             mean_distance_reduction=value,accepted=n,
                             acceptance_definition="joint" if method=="joint" else "class_only"))
    d=pd.DataFrame(rows)
    if extra:
        e=read_csv(extra)
        required={"method","label","budget","n_sources","mean_distance_reduction","accepted","acceptance_definition"}
        if not required<=set(e):raise ValueError(f"Transfer table requires {required}")
        if not (e.n_sources==11).all():raise ValueError("All frontier methods must use the same 11 source beers, including fallbacks.")
        d=pd.concat([d,e],ignore_index=True)
    export_rows("08_transfer_frontier",d)
    palette={"distance":C["gray"],"class_only":C["reject"],"joint":C["teal"]}
    marks={"distance":"s","class_only":"x","joint":"o"}
    fig,axs=plt.subplots(1,2,figsize=(8.6,4.2))
    fig.subplots_adjust(left=.09,right=.965,bottom=.25,top=.76,wspace=.30)
    handles=[]
    for i,(method,g) in enumerate(d.groupby("method",sort=False)):
        g=g.sort_values("budget");color=palette.get(method,C["orange"]);marker=marks.get(method,"D")
        label=g.label.iloc[0]
        line,=axs[0].plot(g.budget,100*g.mean_distance_reduction,color=color,marker=marker,label=label,lw=1.7)
        axs[1].plot(g.budget,g.accepted,color=color,marker=marker,label=label,lw=1.7)
        handles.append(line)
    axs[0].axhline(0,color=C["text"],lw=.9)
    axs[0].set(ylabel="11 款平均目标距离下降 / %")
    axs[1].set(ylabel="输出中判为 Tripel 的源谱数 / 11",ylim=(-.3,11.7),yticks=[0,3,6,9,11])
    for ax in axs:
        ax.set_xscale("log");ax.set_xlim(2.7,245)
        ax.set_xticks([3,5,10,20,40,80,140,225]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{int(x)}"))
        ax.minorticks_off();ax.set_xlabel("可变直接字段预算 k · 对数轴")
        clean(ax,"y")
    panel_title(axs[0],"a","靠近目标需要真实距离改善")
    panel_title(axs[1],"b","Tripel 归类覆盖")
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,.99),ncol=min(4,len(handles)),fontsize=9.5,columnspacing=1.7)
    fig.text(.09,.075,"11 款真实源啤酒 · 未接收候选保留源谱 · 每个标记对应实际求解的预算。",fontsize=9.2,color=C["text"])
    save(fig,OUT/"08_transfer_cost_frontier","稀疏迁移的成本前沿。每条线连接实际测试预算，横轴使用对数坐标；左图始终对全部 11 款源啤酒平均，回退原谱的改善为零。右图保留各方法原有筛选定义：距离目标仅附加检查 Tripel 归类，分类筛选仅要求分类及原有缺陷限制，联合筛选还要求目标距离改善；这些数量不能统称为同一种成功率。新增方法必须使用同一 11 款源谱与同一目标距离度量。")


def certified_sparse_cost(experiment):
    """Compatible solver lower bounds and decoded, fully audited feasible costs."""
    main=read_csv(experiment/"main_comparison.csv")
    policies=read_csv(experiment/"policy_summary.csv")
    checked=read_csv(experiment/"selected_full_checks.csv")
    summary=read_json(experiment/"summary.json")
    completion=read_json(experiment/"COMPLETE.json")
    assert len(main)==11 and main.beer.nunique()==11
    assert (main.strict_lower<=main.strict_upper).all()
    assert (main.strict_distance>=.1-1e-8).all() and (main.strict_stale<=1e-8).all()
    assert (main.strict_observed_fields_changed+main.strict_unknown_fields_completed==main.strict_upper).all()
    checks=checked[(checked.evaluation_policy=="strict")&np.isclose(checked.required_rho,.1)]
    for row in main.itertuples():
        witness=checks[checks.candidate_uid==row.strict_candidate_uid]
        assert len(witness)>=1 and witness.accepted.all()
        assert (witness.raw_fields_changed==row.strict_upper).all()
    main=main.sort_values(["strict_upper","beer"],kind="stable").reset_index(drop=True)
    strict=policies[policies.policy=="strict"].sort_values("rho")
    assert (strict.accepted==11).all()
    export_rows("12_sparse_certified_cost",main)
    export_rows("12_sparse_policy_frontier",policies)
    old_total=int(main.legacy_strict_upper.sum());new_total=int(main.strict_upper.sum())
    saving=1-new_total/old_total
    fig=plt.figure(figsize=(8.6,4.95))
    ax=fig.add_axes([.095,.23,.51,.54]);bx=fig.add_axes([.735,.23,.235,.54])
    yy=np.arange(len(main))
    ax.hlines(yy,main.strict_upper,main.legacy_strict_upper,color=C["line"],lw=1.35,zorder=1)
    ax.hlines(yy,main.strict_lower,main.strict_upper,color=C["teal"],lw=2.0,zorder=2)
    ax.scatter(main.strict_lower,yy,marker="|",s=68,color=C["teal"],linewidths=1.35,zorder=3)
    ax.scatter(main.strict_upper,yy,s=30,c=C["teal"],zorder=4)
    ax.scatter(main.legacy_strict_upper,yy,s=27,marker="s",facecolors="white",edgecolors=C["gray"],linewidths=1.3,zorder=3)
    for i,row in main.iterrows():
        ax.annotate(str(int(row.strict_upper)),(row.strict_upper,i),xytext=(4,-9),textcoords="offset points",
                    va="center",fontsize=9.2,color=C["teal"])
    xmax=max(105,5+int(main.legacy_strict_upper.max()))
    ax.set(xlim=(0,xmax),ylim=(len(main)-.2,-.65),yticks=yy,
           yticklabels=[str(int(b)) for b in main.beer],xticks=np.arange(0,xmax,20),
           xlabel="修改或补全的直接字段数",ylabel="源样本编号")
    clean(ax,"x");ax.tick_params(axis="y",length=0)
    panel_title(ax,"a","同一 10% 改善约束下的成本",y=1.11)
    handles=[Line2D([],[],color=C["teal"],marker="|",lw=1.8,label="下界—可行上界"),
             dot_handle(C["gray"],"旧支持策略",marker="s",filled=False)]
    ax.legend(handles=handles,loc="lower left",bbox_to_anchor=(-.01,1.24),ncol=2,columnspacing=1.3,fontsize=9.5)
    xs=100*strict.rho.to_numpy()
    bx.plot(xs,strict.max_fields,color=C["gray"],marker="s",lw=1.5,ls=(0,(3,2)),label="最大编辑数",zorder=2)
    bx.plot(xs,strict.median_fields,color=C["teal"],marker="o",lw=1.9,label="中位编辑数",zorder=3)
    for x,med,maximum in zip(xs,strict.median_fields,strict.max_fields):
        bx.annotate(f"{med:g}",(x,med),xytext=(0,-15),textcoords="offset points",ha="center",fontsize=9.5,color=C["teal"])
        bx.annotate(f"{maximum:g}",(x,maximum),xytext=(0,7),textcoords="offset points",ha="center",fontsize=9.5,color=C["text"])
    bx.set(xlim=(-2,42),ylim=(7,max(80,int(strict.max_fields.max())+10)),
           xticks=xs,xticklabels=[f"{x:g}" for x in xs],yticks=[20,40,60,80],
           xlabel="要求距离下降 ρ / %",ylabel="可行编辑数")
    clean(bx,"y");panel_title(bx,"b","距离要求与编辑成本",y=1.11)
    bx.legend(loc="lower left",bbox_to_anchor=(-.08,1.24),ncol=2,fontsize=9.5,columnspacing=.85,handlelength=1.5)
    ax.text(.61,.23,f"总成本 {old_total} → {new_total}\n减少 {100*saving:.1f}%",transform=ax.transAxes,
            fontsize=11.2,color=C["teal"],linespacing=1.55)
    fig.text(.095,.074,"a：[L, U] 为同政策数值下界与完整复核可行上界；成本包含缺失字段补全。",fontsize=9.5,color=C["text"])
    fig.text(.095,.029,"b：各阈值均覆盖 11 款源谱；线连接实际求解阈值，全部执行严格缺陷政策。",fontsize=9.5,color=C["text"])
    closed=int((main.strict_lower==main.strict_upper).sum())
    caption=(f"全坐标稀疏迁移的可行成本与求解界。a 在同一冻结分类器、六模型感官接口、化学边界、线性陈旧味守卫、分类间隔 10⁻⁵ 和至少 10% 目标距离改善下比较 11 款源谱。深色横段从同政策汇总的兼容 MIQCP 数值下界 L 延伸至原尺度完整复核可行解 U，实心点与旁侧数字表示 U；空心方块为相同科学约束下重新求解的旧支持策略。实际编辑数为 {int(main.strict_upper.min())}—{int(main.strict_upper.max())}，中位数由 {main.legacy_strict_upper.median():g} 降至 {main.strict_upper.median():g}，总数由 {old_total} 降至 {new_total}（减少 {100*saving:.1f}%）。当前 {closed}/11 个区间闭合；未闭合区间提供可行解与下界，不代表已经证明全局最少。编辑成本包含原观测字段改变和缺失字段补全。两种搜索的计算预算并不相同，图比较同科学约束的结果质量。b 展示严格政策下四个实际距离阈值对应的可行中位与最大编辑数；每个阈值均覆盖全部 11 款。")
    save(fig,OUT/"12_sparse_certified_cost",caption)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-dir",type=Path,default=HERE.parent/"project/MathorCup_A")
    ap.add_argument("--prediction-extra-csv",type=Path)
    ap.add_argument("--modern-dir",type=Path,default=HERE.parent/"experiments/modern_prediction")
    ap.add_argument("--paired-modern-model",default="raw231_y16_no_residual")
    ap.add_argument("--transfer-extra-csv",type=Path)
    ap.add_argument("--semantic-dir",type=Path,default=HERE.parent/"experiments/semantic_auxiliary")
    ap.add_argument("--classification-certificate-csv",type=Path,default=HERE.parent/"experiments/sparse_transfer/classification_geometry_certificates.csv")
    ap.add_argument("--within-style-dir",type=Path,default=HERE.parent/"experiments/within_style_explanation")
    ap.add_argument("--sparse-summary-dir",type=Path,default=HERE.parent/"experiments/sparse_transfer/final_summary")
    args=ap.parse_args();project=args.project_dir.resolve();work=project/"A_work"
    configure(project/"assets/fonts/NotoSansCJKsc-Regular.otf")
    names=read_json(work/"figures/sensory_names_zh.json")
    semantic_dir=args.semantic_dir if args.semantic_dir.is_dir() else None
    prediction(work,args.prediction_extra_csv,semantic_dir,args.modern_dir,args.paired_modern_model)
    missingness(work,args.modern_dir/"missingness");network_modules(work,names);generation(work);transfer(work,args.transfer_extra_csv)
    if semantic_dir is not None:
        semantic(work,semantic_dir)
        semantic_style_controls(semantic_dir)
    if args.classification_certificate_csv.exists():classification_certificate(args.classification_certificate_csv)
    if args.within_style_dir.is_dir():within_style(args.within_style_dir)
    if (args.sparse_summary_dir/"main_comparison.csv").is_file():certified_sparse_cost(args.sparse_summary_dir)
    (OUT/"evidence_manifest.json").write_text(json.dumps(dict(sources=SOURCES,plot_tables=PLOT_DATA),ensure_ascii=False,indent=2))
    print(json.dumps(dict(figures=len(list(OUT.glob("*.svg"))),formats=["svg","pdf","png"],source_files=len(SOURCES),plot_tables=len(PLOT_DATA),out=str(OUT)),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
