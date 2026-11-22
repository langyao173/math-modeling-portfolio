"""A report-sized budget comparison from the saved eleven-source results.

This file only reads saved results; it does not refit or change any model.
"""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mathorcup_sparse_report_mpl")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
AWORK = HERE.parents[1]
font_paths = [AWORK.parent / "assets/fonts/NotoSansCJKsc-Regular.otf",
              AWORK / "package/MathorCup_A/assets/fonts/NotoSansCJKsc-Regular.otf"]
for path in font_paths:
    if path.exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
plt.rcParams.update({"font.size":10.8, "axes.spines.top":False,
    "axes.spines.right":False, "axes.edgecolor":"#8D969F",
    "axes.labelcolor":"#28323C", "text.color":"#28323C",
    "xtick.color":"#4C5863", "ytick.color":"#4C5863",
    "svg.fonttype":"none", "axes.unicode_minus":False})

summary = pd.read_csv(HERE / "pareto_summary.csv").set_index("method")
diagnostics = pd.read_csv(HERE / "selected_diagnostics.csv")
budgets = np.array([3,5,10,20])
families = [
    ("gradient", "静态梯度排序", "#7C8D99", "o"),
    ("full_change", "全维改变量排序", "#87956F", "s"),
    ("forward_schur", "逐步 Schur 选择", "#225D75", "D")]
values = {"allowed_fields":budgets, "n_sources":np.repeat(11,len(budgets))}
fig,ax = plt.subplots(figsize=(6.2,3.8))
for family,label,color,marker in families:
    means = []
    for k in budgets:
        method = f"{family}_k{k}"
        per_case = diagnostics.loc[diagnostics.method.eq(method),"target_distance_reduction"]
        assert len(per_case)==11
        assert np.isclose(per_case.mean(),summary.loc[method,"target_distance_reduction_mean"],atol=1e-12,rtol=0)
        means.append(100*per_case.mean())
    means=np.array(means)
    values[family+"_mean_percent"]=means
    is_forward=family=="forward_schur"
    ax.plot(budgets,means,label=label,color=color,marker=marker,
        lw=2.2 if is_forward else 1.65,ms=5.5 if is_forward else 4.8,
        markeredgewidth=.6,markeredgecolor="white",zorder=4 if is_forward else 3)
    ax.text(20.55,means[-1],f"{means[-1]:.2f}%",color=color,
        fontsize=10.5,fontweight="bold" if is_forward else "normal",va="center")
ax.set(xlim=(2,24),ylim=(0,25),xticks=budgets,yticks=np.arange(0,26,5),
    xlabel="最多允许调整的直接字段数",ylabel="平均目标距离下降 / %")
ax.set_title("有限变量预算下的目标距离下降",loc="left",fontsize=11.5,pad=11,fontweight="bold")
ax.set_axisbelow(True)
ax.grid(axis="y",color="#E3E8EC",lw=.65)
ax.legend(loc="upper left",frameon=False,fontsize=10.1,labelspacing=.55,handlelength=2.3)
fig.subplots_adjust(left=.12,right=.98,bottom=.23,top=.86)
fig.text(.12,.055,"全部11例及零移动回退均计入；指标来自同一冻结模型与验收规则。",
         fontsize=9.1,color="#5F6B75")
fig.savefig(HERE / "sparse_transfer_report.png",dpi=260,bbox_inches="tight",pad_inches=.08)
fig.savefig(HERE / "sparse_transfer_report.svg",bbox_inches="tight",pad_inches=.08)
plt.close(fig)
pd.DataFrame(values).to_csv(HERE / "report_plot_values.csv",index=False)
print("Saved sparse_transfer_report.png and sparse_transfer_report.svg; all 12 means checked against eleven individual cases.")
