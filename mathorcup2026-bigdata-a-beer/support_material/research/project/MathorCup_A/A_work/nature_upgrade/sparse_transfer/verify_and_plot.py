"""Independent exported-profile checks and a compact intervention-budget figure."""
from pathlib import Path
import json, os, argparse
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mathorcup_sparse_mpl")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
AWORK=HERE.parents[1]
data_candidates=[AWORK.parent/"data/A",AWORK/"package/MathorCup_A/data/A"]
default_data=Path(os.environ["MATHORCUP_DATA_DIR"]).expanduser() if os.environ.get("MATHORCUP_DATA_DIR") else next((p for p in data_candidates if (p/"1.csv").exists()),AWORK.parent/"data/A")
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data",type=Path,default=default_data)
data=ap.parse_args().data
source=pd.read_csv(data/"1.csv").set_index("beer")
profiles=pd.read_csv(HERE/"selected_profiles.csv")
diagnostics=pd.read_csv(HERE/"selected_diagnostics.csv")
supports=pd.read_csv(HERE/"selected_supports.csv").set_index(["beer","method"])
summary=pd.read_csv(HERE/"pareto_summary.csv").set_index("method")
derived=["kcalperc","esters_sum","aroma_hops_sum","acids_sum","sulfur_sum"]
internal="X3.methyl.2.butenal"
features=[c for c in source.columns[2:] if c not in derived+[internal]]
defects={"DMS":("log10",19.),"X2.3.butanedione":("identity",.09),
         "X4VG":("log10",.2),"X.E..2.nonenal":("identity",.0001)}
violations=[]
max_energy_error=0.
for i,row in profiles.iterrows():
    src=source.loc[int(row.beer)]
    support=set(json.loads(supports.loc[(row.beer,row.method),"selected_fields"]))
    for col in features:
        if col not in support:
            a,b=src[col],row[col]
            if not ((pd.isna(a) and pd.isna(b)) or np.isclose(a,b,rtol=0,atol=1e-14)):
                violations.append([int(row.beer),row.method,col,"frozen field changed"])
    if not np.isclose(src[internal],row[internal],rtol=0,atol=1e-14,equal_nan=True):
        violations.append([int(row.beer),row.method,internal,"internal standard changed"])
    if diagnostics.loc[i,"nontrivial"]:
        for col,(transform,threshold) in defects.items():
            before,after=src[col],row[col]
            if transform=="log10":before,after=10**before,10**after
            limit=before if np.isfinite(before) else threshold
            if not np.isfinite(after) or after>limit+1e-7:
                violations.append([int(row.beer),row.method,col,"chemical nonworsening violation"])
        caloric=5.523*10**row["ethanol..v.v."]+.4*(10**row["GFS"]+10**row["unfermentables"]+10**row["protein.g.L."])+.0003*10**row["lactic_acid.mg.L."]
        max_energy_error=max(max_energy_error,abs(row.kcalperc-np.log10(caloric)))
for family in ["gradient","full_change","forward_schur"]:
    for beer,g in diagnostics[diagnostics.method.str.startswith(family+"_k")].groupby("beer"):
        g=g.sort_values("allowed_coordinates")
        if np.any(np.diff(g.quadratic_objective)>1e-7):
            violations.append([int(beer),family,"objective","nested budget worsened best objective"])
assert not violations, violations
audit=dict(n_exported_cases=len(profiles),n_nontrivial=int(diagnostics.nontrivial.sum()),
           n_fallback=int((~diagnostics.nontrivial).sum()),
           exported_frozen_fields_verified=True,internal_standard_preserved=True,
           physical_scale_chemical_checks_passed=True,nested_budget_best_objective_monotone=True,
           max_caloric_log_identity_error=float(max_energy_error),violations=violations,
           fallback_missing_defect_count=int(diagnostics.loc[~diagnostics.nontrivial,"unresolved_missing_defects"].sum()))
(HERE/"independent_checks.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False))

font=next((p for p in [AWORK.parent/"assets/fonts/NotoSansCJKsc-Regular.otf",AWORK/"package/MathorCup_A/assets/fonts/NotoSansCJKsc-Regular.otf"] if p.exists()),Path("/missing-font"))
if font.exists():
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"]=font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":"#889099","axes.labelcolor":"#222B34",
                     "text.color":"#222B34","xtick.color":"#535D67","ytick.color":"#535D67",
                     "svg.fonttype":"none"})
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.8,4.5),gridspec_kw={"width_ratios":[1.05,1.2]})
colors={"gradient":"#7F9AAB","full_change":"#A0A99D","forward_schur":"#225D75"}
for family,label in [("gradient","源点静态梯度排序"),("full_change","完整解改变量排序"),("forward_schur","逐步Schur选变量")]:
    ks=np.array([3,5,10,20])
    means=np.array([100*summary.loc[f"{family}_k{k}","target_distance_reduction_mean"] for k in ks])
    quantiles=np.array([100*diagnostics.loc[diagnostics.method==f"{family}_k{k}","target_distance_reduction"].quantile([.25,.75]).to_numpy() for k in ks])
    ax.fill_between(ks,quantiles[:,0],quantiles[:,1],color=colors[family],alpha=.08,lw=0)
    ax.plot(ks,means,"o-",ms=5,lw=2.2 if family=="forward_schur" else 1.5,color=colors[family],label=label)
ax.set(xlim=(2,21),ylim=(0,32),xticks=[3,5,10,20],xlabel="允许调整的直接字段数",ylabel="平均目标距离下降 / %")
ax.set_title("A  少量调整的收益曲线",loc="left",fontweight="bold",pad=13)
ax.grid(axis="y",color="#E8EAED",lw=.7)
ax.legend(frameon=False,loc="upper left",fontsize=9)
ax.text(.02,.02,"阴影：11个源样本的四分位区间",transform=ax.transAxes,fontsize=8.3,color="#626B74")
comparisons=[("author7_add_only","原文7项集合，仅增加"),
             ("gradient_k20","静态梯度20项"),("forward_schur_k20","逐步Schur选20项"),("all_225","全部225项可调")]
for row,(method,label) in enumerate(comparisons):
    yy=3-row
    vals=100*diagnostics.loc[diagnostics.method==method,"target_distance_reduction"].to_numpy()
    offsets=np.linspace(-.12,.12,len(vals))
    bx.scatter(vals,yy+offsets,s=22,color="#BAC4CB",edgecolor="white",linewidth=.5,zorder=2)
    m=100*summary.loc[method,"target_distance_reduction_mean"]
    bx.scatter([m],[yy],s=72,marker="D",color="#315A7A",edgecolor="white",linewidth=.8,zorder=3)
    bx.text(m+2.7,yy+.20,f"{m:.2f}%",fontsize=10,fontweight="bold",color="#315A7A")
bx.set(xlim=(-2,90),ylim=(-.5,3.6),yticks=[3,2,1,0],yticklabels=[x[1] for x in comparisons],xlabel="目标距离下降 / %")
bx.set_title("B  原文变量集与风格迁移目标",loc="left",fontweight="bold",pad=13)
bx.grid(axis="x",color="#E8EAED",lw=.7)
bx.tick_params(axis="y",length=0)
fig.subplots_adjust(left=.07,right=.98,top=.86,bottom=.22,wspace=.62)
fig.text(.07,.07,"所有平均值均以11个源样本为分母；圆点为样本，菱形为均值。原文7项是变量集移植对照，目标为本题风格迁移。",fontsize=8.3,color="#626B74")
fig.text(.07,.025,"原文7项产生8个非零候选，其余右图方案均为11个；仅全维方案有10个被判为Tripel。模型距离不是实酿风味改善比例。",fontsize=8.3,color="#626B74")
fig.savefig(HERE/"sparse_transfer_pareto.svg",bbox_inches="tight")
fig.savefig(HERE/"sparse_transfer_pareto.png",dpi=220,bbox_inches="tight")
plt.close(fig)
print(json.dumps(audit,ensure_ascii=False))
