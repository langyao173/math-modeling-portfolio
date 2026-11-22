"""Retain the reference-component path study, using V for its base-score index."""
import pandas as pd
import v17_figures as old
from v18_figures import ROOT,E,save
def read(_):return pd.read_csv(E/'epoch_attribution_rows.csv.gz')
def export(fig,_):
    for ax in fig.axes:
        ax.set_xlabel(ax.get_xlabel().replace('ΔQ','ΔV'))
        ax.set_ylabel(ax.get_ylabel().replace('ΔQ','ΔV'))
    save(fig,'v18_reference_epoch')
if __name__=='__main__':
    old.read=read;old.save=export;old.epoch()
