"""Development candidate: calibrated expectations with relative agent corrections.

This is a demand/investment module experiment, NOT a replacement national fuel
model or native Simulink result. Frozen V3 cohorts and sources are not modified.
All model selection is repeated using only data at the historical origin.
"""
from __future__ import annotations
import json
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src import v3_annual as frozen

MODES=("increment3","gdp_full","gdp_recent5","intensity3")
TECHS=frozen.TECHS


def _series(data,end):
    a,g,k,u,shares,direct=frozen._observations(data,end)
    return a.reset_index(drop=True),g,k,u


def _fit_growth(values,gdp,mode):
    dx=np.diff(np.log(gdp));dy=np.diff(np.log(values))
    if mode=="increment3":return {"mode":mode,"increment":float(np.diff(values)[-3:].mean())}
    if mode=="intensity3":return {"mode":mode,"beta":1.,"drift":float((dy-dx)[-3:].mean())}
    if mode=="gdp_recent5":dx,dy=dx[-5:],dy[-5:]
    return {"mode":mode,"beta":float(dx@dy/max(dx@dx,1e-12)),"drift":0.}


def _curve(base,basegdp,gdp_path,fit):
    h=np.arange(1,len(gdp_path)+1)
    if fit["mode"]=="increment3":return np.maximum(base+fit["increment"]*h,1e-9)
    return base*(np.asarray(gdp_path)/basegdp)**fit["beta"]*np.exp(fit["drift"]*h)


def _select_growth(values,gdp,years):
    records=[];errors={m:[] for m in MODES}
    # Same inner targets for every candidate; four observations are sufficient
    # for each declared candidate (recent5 uses all available changes if <5).
    for n in range(4,len(values)):
        for h in (1,3):
            if n+h-1>=len(values):continue
            for mode in MODES:
                fit=_fit_growth(values[:n],gdp[:n],mode)
                pred=float(_curve(values[n-1],gdp[n-1],gdp[n:n+h],fit)[-1])
                ape=abs(pred/values[n+h-1]-1)*100;errors[mode].append(ape)
                records.append({"origin_year":int(years[n-1]),"horizon":h,"method":mode,"observed":float(values[n+h-1]),"predicted":pred,"ape_percent":ape})
    score={m:float(np.mean(v)) for m,v in errors.items()}
    mode=min(MODES,key=lambda m:score[m])
    return _fit_growth(values,gdp,mode),score,records


def _fit_basic(data,end):
    a,g,k,service=_series(data,end);gdp=a.gdp_real_index.to_numpy(float);years=a.year.to_numpy(int)
    ufit,uscore,urecords=_select_growth(service,gdp,years)
    efit,escore,erecords=_select_growth(a.electricity_demand_twh.to_numpy(float),gdp,years)
    # ONE pooled nonnegative response coefficient, not seven free preferences.
    # Predict net additions L years after an observed utilisation innovation.
    # Retirement is declared engineering structure, never estimated from net K.
    x=[];y=[];eng=data["config"]["engineering"];lags=np.array(eng["construction_years"][:7],int)
    e=a.electricity_demand_twh.to_numpy(float)
    for t in range(4,len(a)):
        expected_e=float(_curve(e[t-1],gdp[t-1],[gdp[t]],efit)[0])
        for j,L in enumerate(lags):
            if t+L>=len(a):continue
            expected_g=g[t-1,j]*expected_e/e[t-1]
            innovation=(g[t,j]-expected_g)/(8.76*k[t,j])
            base=float(np.diff(k[t-3:t+1,j]).mean())
            net_residual=(k[t+L,j]-k[t+L-1,j]-base)/k[t,j]
            x.append(innovation);y.append(net_residual)
    x=np.array(x);y=np.array(y)
    coefficient=float(np.clip(x@y/(x@x+.05*max(len(x),1)),0,.5)) if len(x) else 0.
    return {"train_end":int(end),"service_fit":ufit,"electricity_fit":efit,"service_cv":uscore,"electricity_cv":escore,
            "capacity_expectation":"recent3_net_increment","utilisation_response_raw":coefficient,"n_utilisation_response_pairs":len(x),
            "response_fit_boundary":"one pooled bounded ridge coefficient; observational delayed net-addition association, not causal return elasticity",
            "demand_selection_records":{"service":urecords,"electricity":erecords}}


def _dispatch(capacity,demand,util,carbon,unitcost,biomass_budget):
    potential=capacity*util*8.76;potential[6]=min(potential[6],biomass_budget)
    generation=potential.copy();generation[:2]=0
    if generation.sum()>demand:generation*=demand/generation.sum()
    variable=np.array([300.,600.,20.,15.,5.,5.,250.]);emission=np.array([.82,.37,0,0,0,0,0]);marginal=variable+carbon*emission
    remaining=max(demand-generation.sum(),0)
    for j in np.argsort(marginal[:2]):generation[j]=min(remaining,potential[j]);remaining-=generation[j]
    price=1200. if remaining>1e-8 else (float(marginal[:2][generation[:2]>1e-8].max()) if (generation[:2]>1e-8).any() else 300.)
    profit=generation*(price-marginal)-capacity*unitcost*.08
    return generation,price,profit,max(remaining,0)


def simulate_candidate(data,params,origin,horizon,gdp_path=None,scenario=None):
    a,g,k,service=_series(data,origin);eng=data["config"]["engineering"];s={"name":"historical","technology_cost_multiplier":1.,"carbon_price_multiplier":1.,"demand_multiplier":1.,"investment_feedback":True,"cost_response":True,"construction_delay_multiplier":1.,"annual_budget_multiplier":1.,**(scenario or {})}
    if gdp_path is None:
        rate=float(np.mean(np.diff(np.log(a.gdp_real_index.to_numpy(float)))[-3:]));gdp_path=a.gdp_real_index.iloc[-1]*np.exp(rate*np.arange(1,horizon+1))
    service_base=_curve(service[-1],float(a.gdp_real_index.iloc[-1]),gdp_path,params["service_fit"])
    electricity_base=_curve(float(a.electricity_demand_twh.iloc[-1]),float(a.gdp_real_index.iloc[-1]),gdp_path,params["electricity_fit"])
    net=np.maximum(np.diff(k[-4:],axis=0).mean(axis=0),0)
    initial=k[-1].copy();capacity=initial.copy();retirement=np.array(eng["retirement_fraction"][:7],float)
    utilisation=np.median(g[-3:]/(k[-3:]*8.76),axis=0)
    lags=np.array(eng["construction_years"][:7],int);delay=np.ceil(lags*s["construction_delay_multiplier"]).astype(int)
    depth=max(15,int(delay.max()));queue=np.zeros((7,depth))
    def reference_capacity(h):return initial+net*h
    def gross_commission(h):return np.maximum(reference_capacity(h)-(1-retirement)*reference_capacity(h-1),0)
    costs=np.array(eng["capital_cost_million_cny"][:7],float)
    actual_cost=costs.copy();actual_cost[4:6]*=s["technology_cost_multiplier"]
    previous_price=450.;reference_previous_price=450.;relative_e=1.;rows=[];actions=[]
    gamma=params.get("utilisation_response",0.) if s["investment_feedback"] else 0.
    # 2024 system initialisation keeps the observed NEA biomass pair separate
    # from the homogeneous Ember history used by the forecasting comparisons.
    biomass_initial=float(g[-1,6])
    if origin==2024:
        b=data["supplement"].loc[data["supplement"].carrier=="biomass"].set_index("metric")
        old=initial[6];initial[6]=float(b.loc["capacity_gw","value"]);capacity[6]=initial[6]
        net[6]*=initial[6]/old;g=g.copy();g[-1,6]=float(b.loc["generation_twh","value"])
        biomass_initial=float(b.loc["generation_twh","value"]);utilisation[6]=biomass_initial/(initial[6]*8.76)
    for j,L in enumerate(lags):
        for slot in range(L):queue[j,slot]=gross_commission(slot+1)[j]
    previous_generation=g[-1].copy();previous_electricity=float(a.electricity_demand_twh.iloc[-1]);previous_gdp=float(a.gdp_real_index.iloc[-1])
    for h in range(1,horizon+1):
        commissioned=queue[:,0].copy();retired=capacity*retirement;capacity=capacity-retired+commissioned
        queue=np.column_stack([queue[:,1:],np.zeros(7)])
        # Partial adjustment acts on deviations from a learned forecast. With
        # no innovation it does not slow down the historical electrification.
        delta_price=previous_price/reference_previous_price-1 if s["cost_response"] else 0.
        relative_e=.5*relative_e+.5*np.exp(-.2*delta_price)
        demand=electricity_base[h-1]*s["demand_multiplier"]*relative_e
        useful=service_base[h-1]*s["demand_multiplier"]
        demand=min(demand,.95*useful/.9)
        bio=biomass_initial*np.exp(.02*h)
        generation,price,profit,unserved=_dispatch(capacity,demand,utilisation,100*s["carbon_price_multiplier"],actual_cost,bio)
        refcap=reference_capacity(h)
        refgen,refprice,refprofit,refunserved=_dispatch(refcap,electricity_base[h-1],utilisation,100.,costs,bio)
        expected_one_step=float(_curve(previous_electricity,previous_gdp,[gdp_path[h-1]],params["electricity_fit"])[0])
        observed_expected_g=previous_generation*expected_one_step/previous_electricity
        util_innovation=(generation-observed_expected_g)/np.maximum(capacity*8.76,1e-9)
        planned=np.array([gross_commission(h+int(delay[j]))[j] for j in range(7)])
        utilisation_adjustment=gamma*capacity*util_innovation
        margin_change=(profit/np.maximum(capacity,1e-9)-refprofit/np.maximum(refcap,1e-9))/np.maximum(costs*.08,1)
        relative_return=np.clip(1+.25*margin_change,0,2) if s["cost_response"] else np.ones(7)
        wanted=np.maximum(planned+utilisation_adjustment,0)*relative_return
        wanted=np.minimum(wanted,planned*2)
        # Budget is a finite, explicit forecast-based envelope, not observed
        # corporate finance. It is not fitted to improve the evaluation score.
        budget=float(planned@costs)*1.15*s["annual_budget_multiplier"]
        orders=wanted*min(1,budget/max(float(wanted@actual_cost),1e-9))
        for j,L in enumerate(delay):queue[j,L-1]+=orders[j]
        row={"year":origin+h,"origin_year":origin,"scenario":s["name"],"electricity_demand_twh":demand,"useful_service_demand_twh":useful,"electricity_unserved_twh":unserved,"expectation_electricity_twh":electricity_base[h-1],"relative_electricity_choice":relative_e,"market_price_cny_mwh":price,"investment_spend_million_cny":float(orders@actual_cost),"investment_budget_million_cny":budget,"utilisation_response":gamma}
        row.update({tech+"_capacity_gw":float(capacity[j]) for j,tech in enumerate(TECHS)})
        rows.append(row)
        for j,tech in enumerate(TECHS):actions.append({"year":origin+h,"scenario":s["name"],"agent":tech,"order_gw":orders[j],"commissioned_gw":commissioned[j],"retired_gw":retired[j],"reference_gross_order_gw":planned[j],"utilisation_innovation":util_innovation[j],"calibrated_utilisation_correction_gw":utilisation_adjustment[j],"relative_profit_change":margin_change[j],"capital_spend_million_cny":orders[j]*actual_cost[j]})
        previous_price=price;reference_previous_price=refprice
        previous_generation=generation.copy();previous_electricity=demand;previous_gdp=gdp_path[h-1]
    return pd.DataFrame(rows),pd.DataFrame(actions)


def calibrate_candidate(data,end):
    params=_fit_basic(data,end);scores={"zero":[],"learned":[]};records=[]
    observed=data["power"].pivot(index="year",columns="technology",values="capacity_gw")
    for origin in range(2015,end):
        basic=_fit_basic(data,origin)
        for h in (1,3):
            if origin+h>end:continue
            gdp=data["annual"].set_index("year").loc[list(range(origin+1,origin+h+1)),"gdp_real_index"].to_numpy(float)
            for method in ("zero","learned"):
                local={**basic,"utilisation_response":0. if method=="zero" else basic["utilisation_response_raw"]}
                pred,_=simulate_candidate(data,local,origin,h,gdp)
                for tech in TECHS:
                    actual=float(observed.loc[origin+h,tech]);value=float(pred.iloc[-1][tech+"_capacity_gw"]);ape=abs(value/actual-1)*100
                    scores[method].append(ape);records.append({"origin_year":origin,"horizon":h,"target":tech+"_capacity_gw","method":method,"ape_percent":ape})
    average={k:float(np.mean(v)) for k,v in scores.items()}
    selected="learned" if average["learned"]<average["zero"]-1e-9 else "zero"
    params.update(utilisation_response=params["utilisation_response_raw"] if selected=="learned" else 0.,utilisation_selected=selected,utilisation_cv_mape=average,utilisation_cv_records=records)
    return params


def run(root):
    root=Path(root);data=frozen.load_data(root);out=root/"outputs/v3";out.mkdir(parents=True,exist_ok=True)
    records=[];selections=[];selection_records=[];gdp=data["annual"].set_index("year").gdp_real_index;capacity=data["power"].pivot(index="year",columns="technology",values="capacity_gw")
    targets=[tech+"_capacity_gw" for tech in TECHS]+["electricity_demand_twh"]
    for origin in range(2017,2024):
        params=calibrate_candidate(data,origin)
        selections.append({"origin_year":origin,"service_method":params["service_fit"]["mode"],"electricity_method":params["electricity_fit"]["mode"],"utilisation_response_raw":params["utilisation_response_raw"],"utilisation_response_selected":params["utilisation_response"],"utilisation_selected":params["utilisation_selected"],"gamma_cv_zero":params["utilisation_cv_mape"]["zero"],"gamma_cv_learned":params["utilisation_cv_mape"]["learned"],"n_utilisation_pairs":params["n_utilisation_response_pairs"]})
        for kind,rows in params["demand_selection_records"].items():selection_records.extend({"outer_origin":origin,"target":kind,**r} for r in rows)
        selection_records.extend({"outer_origin":origin,**r} for r in params["utilisation_cv_records"])
        for horizon in (1,3):
            if origin+horizon>2024:continue
            predicted,_=simulate_candidate(data,params,origin,horizon,gdp.loc[list(range(origin+1,origin+horizon+1))].to_numpy(float))
            for target in targets:
                actual=float(data["annual"].set_index("year").loc[origin+horizon,target]) if target=="electricity_demand_twh" else float(capacity.loc[origin+horizon,target.removesuffix("_capacity_gw")])
                value=float(predicted.iloc[-1][target]);records.append({"origin_year":origin,"horizon":horizon,"target_year":origin+horizon,"target":target,"method":"expectation_anchored_candidate","observed":actual,"predicted":value,"ape_percent":abs(value/actual-1)*100})
    with zipfile.ZipFile(out/"annual_baseline_before_calibrated_expectations.zip") as archive:
        old=pd.read_csv(io.BytesIO(archive.read("outputs/v3/historical_rolling.csv")))
    comparison=pd.concat([old,pd.DataFrame(records)],ignore_index=True)
    comparison.to_csv(out/"calibration_candidate_comparisons.csv",index=False)
    metrics=comparison.groupby(["target","method","horizon"]).ape_percent.agg(["mean","count"]).reset_index();metrics.to_csv(out/"calibration_candidate_metrics.csv",index=False)
    pd.DataFrame(selections).to_csv(out/"calibration_candidate_selections.csv",index=False);pd.DataFrame(selection_records).to_csv(out/"calibration_candidate_inner_cv.csv",index=False)
    params=calibrate_candidate(data,2024);frames=[];actions=[]
    scenarios={"reference":{},"cost_level_0.65":{"technology_cost_multiplier":.65},"demand_1.2":{"demand_multiplier":1.2},"demand_1.2_feedback_off":{"demand_multiplier":1.2,"investment_feedback":False},"cost_0.65_delay2":{"technology_cost_multiplier":.65,"construction_delay_multiplier":2},"budget_0.6":{"annual_budget_multiplier":.6}}
    for name,s in scenarios.items():
        frame,action=simulate_candidate(data,params,2024,12,scenario={"name":name,**s});frames.append(frame);actions.append(action)
    pd.concat(frames).to_csv(out/"calibration_candidate_mechanisms.csv",index=False);pd.concat(actions).to_csv(out/"calibration_candidate_actions.csv",index=False)
    summary={"status":"EXPLORATORY_CALIBRATION_MODULE_CANDIDATE","cohort":{"h1_origins":list(range(2017,2024)),"h3_origins":list(range(2017,2022)),"targets":targets},"forecast_information":"given realised GDP; future energy/capacity/generation observations not supplied","module_scope":"Demand expectations and generation-investment construction module; fuel inventories, national emissions and native Simulink are NOT implemented by this candidate","initial_pipeline":"Forecast-based committed commissioning, not observed projects; gross flow is an accounting identity from a baseline net expectation and declared retirement, not a fitted free parameter","engineering":{"retirement":data["config"]["engineering"]["retirement_fraction"][:7],"construction_years":data["config"]["engineering"]["construction_years"][:7],"budget_headroom":1.15,"relative_profit_response":.25,"relative_demand_response":.2,"partial_adjustment_of_innovations":.5},"selection_scope":"Every outer origin reselects using earlier 1/3-year rolling targets only; current candidate design is exploratory after the frozen failures were seen","final_2024_parameters":params,"diagnosis":{"demand":"Independent useful-service GDP extrapolation and absolute target-share adjustment jointly created consistent downward electric-demand drift; candidate adjusts deviations from a learned electric-demand expectation instead","solar":"Early short-history percentage growth and inferred pipeline perpetuated solar growth; initial commitments now follow the declared net-increment expectation and replacement identity; relative response is separately selected"}}
    (out/"calibration_candidate_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),"utf-8")
    print(metrics.loc[metrics.target.isin(["solar_capacity_gw","electricity_demand_twh"])].to_string(index=False))


if __name__=="__main__":run(Path(__file__).resolve().parents[1])
