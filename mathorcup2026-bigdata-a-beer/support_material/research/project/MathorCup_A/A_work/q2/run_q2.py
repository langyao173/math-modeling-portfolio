"""MathorCup 2026 A, question 2: reproducible chemical graph and driver analysis.




This module never imports the supplied GBM importances or SHAP values as
new experimental evidence. Original data files are read only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm, rankdata, spearmanr, t
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

SEED = 20261023
NON_MOLECULE = {
    'pH', 'color.EBC.', 'betaglucan.mg.L.', 'protein.g.L.', 'bitterness',
    'GFS', 'unfermentables', 'kcalperc', 'esters_sum', 'aroma_hops_sum',
    'acids_sum', 'sulfur_sum', 'X3.methyl.2.butenal',
}
ALPHAS = np.array([0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.])
GAMMAS = np.array([0.03, 0.1, 0.3, 1., 3.])


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')


def rank_correlation(X):
    ranks = rankdata(X, axis=0, method='average')
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.corrcoef(ranks, rowvar=False)
    R = np.nan_to_num(R)
    np.fill_diagonal(R, 1.)
    return np.clip(R, -1, 1)


class RankNormalizer:
    """Training-only median imputation and midpoint normal-score transform."""
    def fit(self, X):
        self.median_ = np.nanmedian(X, axis=0)
        self.median_ = np.nan_to_num(self.median_)
        Xm = np.where(np.isfinite(X), X, self.median_)
        self.sorted_ = np.sort(Xm, axis=0)
        self.mean_ = np.zeros(X.shape[1])
        self.scale_ = np.ones(X.shape[1])
        Z = self.transform(X)
        self.mean_ = Z.mean(0)
        self.scale_ = Z.std(0)
        self.scale_[self.scale_ < 1e-12] = 1.
        self.imputed_ = Xm
        return self

    def transform(self, X):
        Xm = np.where(np.isfinite(X), X, self.median_)
        n = len(self.sorted_)
        out = np.empty_like(Xm, dtype=float)
        for j in range(Xm.shape[1]):
            a = self.sorted_[:, j]
            left = np.searchsorted(a, Xm[:, j], side='left')
            right = np.searchsorted(a, Xm[:, j], side='right')
            u = (left + right + 1.) / (2. * (n + 1.))
            out[:, j] = norm.ppf(u)
        return (out - self.mean_) / self.scale_


def signed_laplacian(R, threshold=0.6):
    A = np.where(np.abs(R) >= threshold, R, 0.).copy()
    np.fill_diagonal(A, 0.)
    d = np.abs(A).sum(0)
    inv = np.divide(1., np.sqrt(d), out=np.zeros_like(d), where=d > 0)
    L = np.diag((d > 0).astype(float)) - inv[:, None] * A * inv[None, :]
    return L


def prepared(X, Y, R=None):
    trans = RankNormalizer().fit(X)
    Z = trans.transform(X)
    mu = Y.mean(0)
    sd = Y.std(0)
    sd[sd < 1e-12] = 1.
    T = (Y - mu) / sd
    R = rank_correlation(trans.imputed_) if R is None else R
    return dict(trans=trans, Z=Z, mu=mu, sd=sd, T=T, R=R,
                XtX=Z.T @ Z / len(Z), XtY=Z.T @ T / len(Z),
                L=signed_laplacian(R))


def solve_coef(prep, alpha, gamma):
    A = prep['XtX'] + alpha * np.eye(prep['XtX'].shape[0]) + gamma * prep['L']
    return cho_solve(cho_factor(A, lower=True, check_finite=False),
                     prep['XtY'], check_finite=False)


def predict(prep, B, X):
    return (prep['trans'].transform(X) @ B) * prep['sd'] + prep['mu']


def choose_params(X, Y, seed, rows_out, label):
    candidates = [(float(a), 0.) for a in ALPHAS]
    candidates += [(float(a), float(g)) for a in ALPHAS for g in GAMMAS]
    errors = np.zeros(len(candidates))
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)
    for train, valid in inner.split(X):
        p = prepared(X[train], Y[train])
        Zv = p['trans'].transform(X[valid])
        Tv = (Y[valid] - p['mu']) / p['sd']
        for k, (a, g) in enumerate(candidates):
            B = solve_coef(p, a, g)
            errors[k] += np.mean((Tv - Zv @ B) ** 2) / 3.
    for (a, g), err in zip(candidates, errors):
        rows_out.append(dict(fold=label, alpha=a, gamma=g, inner_mean_MSE=float(err)))
    kr = int(np.argmin(errors[:len(ALPHAS)]))
    kg = len(ALPHAS) + int(np.argmin(errors[len(ALPHAS):]))
    return candidates[kr], candidates[kg]


def greedy_modularity(W):
    """Deterministic weighted greedy modularity; no optional graph package."""
    W = W.copy()
    np.fill_diagonal(W, 0.)
    m = W.sum() / 2.
    n = len(W)
    if m == 0:
        return np.arange(n) + 1
    groups = [{i} for i in range(n)]
    degrees = W.sum(1)
    active = np.ones(n, dtype=bool)
    while True:
        gain = W / m - degrees[:, None] * degrees[None, :] / (2. * m * m)
        gain[~active, :] = -np.inf
        gain[:, ~active] = -np.inf
        gain[np.tril_indices(n)] = -np.inf
        i, j = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[i, j] <= 1e-12:
            break
        groups[i] |= groups[j]
        groups[j] = set()
        active[j] = False
        W[i, :] += W[j, :]
        W[:, i] += W[:, j]
        W[j, :] = 0.
        W[:, j] = 0.
        W[i, i] = 0.
        degrees[i] += degrees[j]
        degrees[j] = 0.
    groups = sorted([g for g in groups if g], key=lambda x: (-len(x), min(x)))
    labels = np.empty(n, dtype=int)
    for k, group in enumerate(groups):
        labels[list(group)] = k + 1
    return labels


def bh_adjust(p):
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.)
    return out


def partial_style_corr(X, Y, groups):
    rx = rankdata(X, axis=0)
    ry = rankdata(Y, axis=0)
    for style in np.unique(groups):
        ix = groups == style
        rx[ix] -= rx[ix].mean(0)
        ry[ix] -= ry[ix].mean(0)
    denom = np.sqrt((rx * rx).sum(0)[:, None] * (ry * ry).sum(0)[None, :])
    return np.divide(rx.T @ ry, denom, out=np.zeros((X.shape[1], Y.shape[1])), where=denom > 0)


def corr_xy(X, Y):
    a = rankdata(X, axis=0)
    b = rankdata(Y, axis=0)
    a -= a.mean(0)
    b -= b.mean(0)
    den = np.sqrt((a*a).sum(0)[:, None] * (b*b).sum(0)[None, :])
    return np.divide(a.T @ b, den, out=np.zeros((X.shape[1], Y.shape[1])), where=den > 0)


def top3(score, eligible):
    out = np.argsort(-np.abs(score[eligible]), axis=0, kind='stable')[:3]
    return eligible[out]


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.data)
    xd = pd.read_csv(src/'1.csv').sort_values('beer').reset_index(drop=True)
    yd = pd.read_csv(src/'4.csv')
    assert xd.beer.nunique() == 250 and yd.beer.nunique() == 250
    yd = yd.set_index('beer').loc[xd.beer].reset_index()
    features = xd.columns[3:].tolist()
    targets = yd.columns[3:].tolist()
    X, Y = xd[features].to_numpy(float), yd[targets].to_numpy(float)
    styles = xd.tasting_category_fine.to_numpy()
    Rfile = pd.read_csv(src/'2.csv', index_col=0)
    R = Rfile.to_numpy(float)
    pearson = xd[features].corr('pearson').to_numpy()
    spearman = xd[features].corr('spearman').to_numpy()
    assert np.max(np.abs(spearman - R)) < 1e-12
    meta = pd.read_excel(src/'8.xlsx', sheet_name='Sensory thresholds').set_index('field_name_R')
    names = pd.read_excel(src/'8.xlsx', sheet_name='Sensory names')
    names = names[names.dataset == 'Trained panel'].set_index('R_name')
    eligible = np.array([i for i, f in enumerate(features) if f not in NON_MOLECULE])
    lookup = pd.DataFrame({'field':features, 'graph_name':Rfile.columns,
        'single_compound_candidate':[f not in NON_MOLECULE for f in features],
        'origin':[meta.loc[f,'origin_paper'] for f in features],
        'unit':[meta.loc[f,'unit'] for f in features],
        'threshold_median':[meta.loc[f,'threshold_median'] for f in features]})
    lookup.to_csv(out/'chemical_dictionary.csv',index=False,encoding='utf-8-sig')

    # The provided all-data graph is descriptive, never used across a held-out fold.
    nvalid = (~np.isnan(X)).astype(int).T @ (~np.isnan(X)).astype(int)
    ii,jj = np.triu_indices(len(features),1)
    rho = R[ii,jj]
    df = nvalid[ii,jj] - 2
    stat = np.abs(rho) * np.sqrt(df / np.maximum(1-rho*rho,1e-15))
    qvalues = bh_adjust(2*t.sf(stat,df))
    graph_rows=[]
    A=np.where(abs(R)>=0.6,R,0.);np.fill_diagonal(A,0.)
    labels=greedy_modularity(abs(A))
    node=pd.DataFrame({'field':features,'name':Rfile.columns,'origin':lookup.origin,
        'module':labels,'degree':(A!=0).sum(1),'weighted_degree':abs(A).sum(1),
        'single_compound_candidate':lookup.single_compound_candidate})
    node.to_csv(out/'graph_nodes.csv',index=False,encoding='utf-8-sig')
    for th in [0.5,0.6,0.7,0.8]:
        W=np.where(abs(R)>=th,abs(R),0.);np.fill_diagonal(W,0.)
        ncomp, comp=connected_components(W,directed=False)
        labs=greedy_modularity(W)
        degrees=W.sum(1);m=W.sum()/2
        modularity=sum(W[np.ix_(labs==k,labs==k)].sum()/(2*m)-(degrees[labs==k].sum()/(2*m))**2 for k in np.unique(labs))
        graph_rows.append(dict(threshold=th,nodes=len(R),edges=int((W>0).sum()/2),
            negative_edges=int((np.triu((R<=-th),1)).sum()),components=int(ncomp),
            largest_component=int(np.bincount(comp).max()),modules=int(len(np.unique(labs))),
            modularity=float(modularity)))
    pd.DataFrame(graph_rows).to_csv(out/'graph_threshold_sensitivity.csv',index=False)

    # Shared five outer folds: no style label, identifier, supplied SHAP or y-derived feature.
    cv = KFold(n_splits=5,shuffle=True,random_state=SEED)
    oof={'rank_ridge':np.zeros_like(Y),'signed_graph_ridge':np.zeros_like(Y),
         'training_mean':np.zeros_like(Y)}
    params=[]; searchrows=[]; fold_id=np.empty(len(X),int)
    for k,(tr,te) in enumerate(cv.split(X)):
        begin=time.time()
        pr,pg=choose_params(X[tr],Y[tr],SEED+k,searchrows,f'outer_{k}')
        p=prepared(X[tr],Y[tr])
        for name,(a,g) in [('rank_ridge',pr),('signed_graph_ridge',pg)]:
            B=solve_coef(p,a,g)
            oof[name][te]=predict(p,B,X[te])
            params.append(dict(fold=k,model=name,alpha=a,gamma=g))
        oof['training_mean'][te]=p['mu']
        fold_id[te]=k
        print('outer',k,'ridge',pr,'graph',pg,'seconds',round(time.time()-begin,1),flush=True)
    pd.DataFrame(params).to_csv(out/'nested_cv_parameters.csv',index=False)
    metrics=[]
    for model,pred in oof.items():
        for j,targ in enumerate(targets):
            metrics.append(dict(model=model,target=targ,RMSE=float(np.sqrt(np.mean((Y[:,j]-pred[:,j])**2))),
                R2=float(r2_score(Y[:,j],pred[:,j])),
                Spearman=float(spearmanr(Y[:,j],pred[:,j]).statistic)))
        pd.DataFrame(dict(beer=xd.beer,fold=fold_id,**{t:pred[:,j] for j,t in enumerate(targets)})).to_csv(out/f'oof_{model}.csv',index=False)
    metricdf=pd.DataFrame(metrics)
    metricdf.to_csv(out/'nested_cv_metrics.csv',index=False)
    pr,pg=choose_params(X,Y,SEED+100,searchrows,'full_training')
    pd.DataFrame(searchrows).to_csv(out/'nested_cv_search.csv',index=False)
    # Select the simpler family by nested OOF aggregate MSE, not full-data fit.
    mse_r=np.mean((oof['rank_ridge']-Y)**2)
    mse_g=np.mean((oof['signed_graph_ridge']-Y)**2)
    family='signed_graph_ridge' if mse_g < mse_r else 'rank_ridge'
    selected=pg if family=='signed_graph_ridge' else pr
    p=prepared(X,Y,R=R)
    B=solve_coef(p,*selected)
    B_r=solve_coef(p,*pr)
    B_g=solve_coef(p,*pg)
    corr=corr_xy(p['trans'].imputed_,Y)
    pcorr=partial_style_corr(p['trans'].imputed_,Y,styles)
    for name,arr in [('coefficients',B),('rank_ridge_coefficients',B_r),('graph_ridge_coefficients',B_g),
        ('spearman_associations',corr),('within_style_rank_associations',pcorr)]:
        pd.DataFrame(arr,index=features,columns=targets).to_csv(out/f'{name}.csv')

    # Fully refit preprocessing and graph in each bootstrap; hyperparameters held
    # at their full-training CV values. This is descriptive selection stability.
    rng=np.random.default_rng(SEED)
    boots=np.empty((args.bootstrap,len(features),len(targets)),dtype=np.float32)
    counts=np.zeros_like(B)
    strat_counts=np.zeros_like(B)
    edge_counts=np.zeros(len(ii),int)
    strata=[np.where(styles==s)[0] for s in np.unique(styles)]
    for b in range(args.bootstrap+args.stratified):
        is_strat=b>=args.bootstrap
        ix=np.concatenate([rng.choice(a,len(a),replace=True) for a in strata]) if is_strat else rng.integers(0,len(X),len(X))
        pb=prepared(X[ix],Y[ix])
        Bb=solve_coef(pb,*selected)
        tops=top3(Bb,eligible)
        target_counts=strat_counts if is_strat else counts
        for j in range(len(targets)):
            target_counts[tops[:,j],j]+=1
        if not is_strat:
            boots[b]=Bb
            edge_counts+=abs(pb['R'][ii,jj])>=0.6
        if (b+1)%100==0:
            print('bootstrap',b+1,'of',args.bootstrap+args.stratified,flush=True)
    counts/=args.bootstrap
    strat_counts/=args.stratified
    lower,upper=np.quantile(boots,[0.025,0.975],axis=0)
    sign_agreement=np.mean(np.sign(boots)==np.sign(B)[None,:,:],axis=0)
    stable_abs=np.median(abs(boots),axis=0)
    np.savez_compressed(out/'bootstrap_coefficients.npz',coefficients=boots,
        features=np.array(features),targets=np.array(targets),seed=SEED)

    edges=pd.DataFrame({'source':[features[i] for i in ii],'target':[features[j] for j in jj],
        'rho':rho,'pairwise_n':nvalid[ii,jj],'approx_BH_q':qvalues,
        'bootstrap_edge_frequency':edge_counts/args.bootstrap})
    edges=edges[abs(edges.rho)>=0.6].sort_values('rho',key=abs,ascending=False)
    edges.to_csv(out/'graph_edges.csv',index=False,encoding='utf-8-sig')

    # Nonlinear alternative for method agreement, not a new held-out prediction claim.
    et_import=np.zeros_like(B)
    Z=p['Z']
    for j,targ in enumerate(targets):
        et=ExtraTreesRegressor(n_estimators=160,max_features=0.7,min_samples_leaf=4,
            random_state=SEED+j,n_jobs=1).fit(Z,Y[:,j])
        et_import[:,j]=et.feature_importances_
        if (j+1)%10==0:print('nonlinear comparison',j+1,'of',len(targets),flush=True)
    pd.DataFrame(et_import,index=features,columns=targets).to_csv(out/'extra_trees_importances.csv')

    # Local sensitivity to imputation on the identical rows/columns.
    # KNN distances are computed after observed-value scaling to avoid raw-unit effects.
    observed_mu=np.nanmean(X,axis=0);observed_sd=np.nanstd(X,axis=0)
    observed_sd[observed_sd<1e-12]=1.
    Xknn=KNNImputer(n_neighbors=5,weights='distance').fit_transform((X-observed_mu)/observed_sd)
    Xknn=Xknn*observed_sd+observed_mu
    pknn=prepared(Xknn,Y)
    Bknn=solve_coef(pknn,*selected)
    cc=~np.isnan(X).any(1)
    pcc=prepared(X[cc],Y[cc])
    Bcc=solve_coef(pcc,*selected)

    methods={'coefficient':B,'spearman':corr,'extra_trees':et_import,
             'ridge':B_r,'signed_graph':B_g,'within_style':pcorr,'KNN':Bknn,'complete_case':Bcc}
    topmethods={k:top3(v,eligible) for k,v in methods.items()}
    primary=np.zeros((3,len(targets)),int)
    rows=[];agreement=[];allrows=[]
    for j,targ in enumerate(targets):
        # Stability-first tie breaking by bootstrap median absolute coefficient.
        order=np.lexsort((-stable_abs[eligible,j],-counts[eligible,j]))
        primary[:,j]=eligible[order[:3]]
        primary_set=set(primary[:,j])
        flags=[]
        for rank,i in enumerate(primary[:,j],1):
            same_module=[features[k] for k in eligible if labels[k]==labels[i] and k!=i]
            near=np.argsort(-abs(R[i]))
            neighbors=[features[k] for k in near if k!=i and abs(R[i,k])>=0.6][:5]
            row=dict(target=targ,target_name=names.loc[targ,'CleanName_PanelFeedback'],rank=rank,
                field=features[i],chemical_name=str(Rfile.columns[i]),origin=lookup.origin.iloc[i],
                coefficient=float(B[i,j]),bootstrap_top3_frequency=float(counts[i,j]),
                stratified_top3_frequency=float(strat_counts[i,j]),
                coefficient_025=float(lower[i,j]),coefficient_975=float(upper[i,j]),
                sign_agreement=float(sign_agreement[i,j]),spearman_rho=float(corr[i,j]),
                within_style_rho=float(pcorr[i,j]),graph_module=int(labels[i]),
                same_module_compounds=int(len(same_module)+1),
                correlated_neighbors=';'.join(neighbors),
                origin_assignment='attachment8_metadata',
                stability_tier='high' if counts[i,j]>=0.8 else ('moderate' if counts[i,j]>=0.5 else 'low'))
            rows.append(row)
        for name,ti in topmethods.items():
            other=set(ti[:,j]);jac=len(primary_set&other)/len(primary_set|other)
            agreement.append(dict(target=targ,method=name,Jaccard=float(jac),
                top3=';'.join(features[k] for k in ti[:,j])))
        for i in range(len(features)):
            allrows.append(dict(target=targ,field=features[i],coefficient=float(B[i,j]),
                bootstrap_top3_frequency=float(counts[i,j]),
                stratified_top3_frequency=float(strat_counts[i,j]),
                bootstrap_sign_agreement=float(sign_agreement[i,j]),
                bootstrap_coef_025=float(lower[i,j]),bootstrap_coef_975=float(upper[i,j]),
                spearman_rho=float(corr[i,j]),within_style_rho=float(pcorr[i,j]),
                graph_module=int(labels[i]),single_compound_candidate=bool(i in eligible)))
    topdf=pd.DataFrame(rows)
    topdf.to_csv(out/'top3_all_50_sensory_dimensions.csv',index=False,encoding='utf-8-sig')
    topdf[topdf.target.str.startswith(('A_','F_'))].to_csv(out/'top3_core_30_aroma_flavour.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(agreement).to_csv(out/'method_and_imputation_agreement.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(allrows).to_csv(out/'all_factor_evidence.csv',index=False,encoding='utf-8-sig')
    # Paired bootstrap of held-out per-beer loss for descriptive uncertainty.
    loss_r=((oof['rank_ridge']-Y)**2).mean(1)
    loss_g=((oof['signed_graph_ridge']-Y)**2).mean(1)
    delta=np.mean(loss_g-loss_r)
    draws=(loss_g-loss_r)[rng.integers(0,len(X),(2000,len(X)))].mean(1)
    ci=np.quantile(draws,[0.025,0.975])
    summary={
        'seed':SEED,'n_samples':len(X),'n_features':len(features),
        'n_single_compound_candidates':len(eligible),'n_sensory_targets':len(targets),
        'n_core_aroma_flavour_targets':sum(t.startswith(('A_','F_')) for t in targets),
        'n_missing_chemistry_values':int(np.isnan(X).sum()),'n_complete_cases':int(cc.sum()),
        'spearman_matrix_max_absolute_error':float(np.max(abs(spearman-R))),
        'pearson_matrix_max_absolute_error':float(np.max(abs(pearson-R))),
        'graph_at_0_6':graph_rows[1],
        'nested_rank_ridge_mean_R2':float(metricdf.query("model == 'rank_ridge'").R2.mean()),
        'nested_graph_mean_R2':float(metricdf.query("model == 'signed_graph_ridge'").R2.mean()),
        'nested_rank_ridge_mean_MSE':float(mse_r),'nested_graph_mean_MSE':float(mse_g),
        'graph_minus_ridge_MSE':float(delta),'paired_beer_bootstrap_delta_CI95':ci.tolist(),
        'CI_note':'Held-out per-beer paired loss bootstrap; does not account for fold-fit dependence or hyperparameter search uncertainty.',
        'selected_family':family,'full_model_alpha':selected[0],'full_model_gamma':selected[1],
        'full_rank_ridge_parameters':pr,'full_signed_graph_parameters':pg,
        'ordinary_bootstrap_replicates':args.bootstrap,'style_stratified_bootstrap_replicates':args.stratified,
        'n_top3_entries_high_stability':int((topdf.bootstrap_top3_frequency>=0.8).sum()),
        'n_top3_entries_frequency_at_least_0_5':int((topdf.bootstrap_top3_frequency>=0.5).sum()),
        'mean_top3_frequency':float(topdf.bootstrap_top3_frequency.mean()),
        'top3_policy':'Rank candidates by ordinary-bootstrap top-3 inclusion frequency; ties by bootstrap median absolute coefficient. Scores calculated on all-data fit; held-out prediction scores kept separate.',
        'excluded_single_compound_candidates':sorted(NON_MOLECULE),
        'source_graph_use':'Attachment 2 for full-data descriptive graph; training-only recomputed graph for all nested validation folds.',
        'model_preprocessing':'Training-only median imputation; midpoint normal-score transform; training standardization of X and y.',
    }
    jdump(summary,out/'summary.json')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=str(Path(__file__).resolve().parent.parent/'data'/'A'),
                        help='Input attachment directory; defaults to ../data/A beside the q2 directory.')
    parser.add_argument('--output',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--bootstrap',type=int,default=400)
    parser.add_argument('--stratified',type=int,default=200)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        main(args)
