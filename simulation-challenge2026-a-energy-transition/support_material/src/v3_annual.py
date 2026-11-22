"""Annual stock-flow ABM. Historic data identify low-dimensional reduced rules.

Construction, retirement, cost preferences, useful-service efficiencies and the
annual flexibility envelope are engineering assumptions, not inferred causality.
The same numeric step is implemented inside a closed-loop native Simulink model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat

TECHS = ("coal", "gas", "nuclear", "hydro", "wind", "solar", "biomass")
ASSETS = (*TECHS, "battery", "pumped", "fuel_coal", "fuel_oil", "fuel_gas")
FUELS = ("coal", "oil", "gas")
TWH_PER_MTCE = 29.3076 / 3.6
NSTATE = 217
QDEPTH = 15
QEND = 20 + 12 * QDEPTH
# State: service0; user shares1:5; nine GW assets5:14; domestic fuel14:17;
# stocks17:20; twelve-by-fifteen queues20:200; shortage200:204;
# utilisation204:211; curtailment211; carbon212; imports213:216; market price216.


def load_data(root):
    root = Path(root)
    annual = pd.read_csv(root / "data/annual_china.csv")
    annual = annual.dropna(subset=["energy_total_mtce", "nonfossil_share", "electricity_demand_twh", "coal_co2_mt"]).copy()
    power = pd.read_csv(root / "data/ember_power_china.csv")
    power.technology = power.technology.replace({"bioenergy": "biomass"})
    power = power.loc[power.technology.isin(TECHS) & power.year.isin(annual.year)].copy()
    supplement = pd.read_csv(root / "data/supplement_carriers_2024.csv")
    return {"annual": annual, "power": power,
            "supplement": supplement,
            "production": pd.read_csv(root / "data/energy_production_china.csv"),
            "trade": pd.read_csv(root / "data/fuel_supply_trade.csv"),
            "config": json.loads((root / "config/v3_transition.json").read_text("utf-8")),
            "source_root": str(root)}


def _observations(data, end):
    a = data["annual"].loc[data["annual"].year <= end].copy().sort_values("year")
    p = data["power"].loc[data["power"].year <= end]
    g = p.pivot(index="year", columns="technology", values="generation_twh").reindex(a.year)[list(TECHS)].to_numpy(float)
    k = p.pivot(index="year", columns="technology", values="capacity_gw").reindex(a.year)[list(TECHS)].to_numpy(float)
    fossil = np.column_stack([a.energy_total_mtce * a[f + "_share"] for f in FUELS])
    power_fuel = np.column_stack([g[:, 0] / (.4 * TWH_PER_MTCE), np.zeros(len(a)), g[:, 1] / (.55 * TWH_PER_MTCE)])
    direct = fossil - power_fuel
    if (direct < 0).any():
        raise ValueError("Observed national fuel cannot cover reconstructed power fuel")
    efficiencies = np.array(data["config"]["engineering"]["direct_use_efficiency"])
    service_parts = np.column_stack([direct * TWH_PER_MTCE, a.electricity_demand_twh]) * efficiencies
    service = service_parts.sum(axis=1)
    return a, g, k, service, service_parts / service[:, None], direct


def _growth_fit(gdp, demand, mode):
    x, y = np.diff(np.log(gdp)), np.diff(np.log(demand))
    if mode == "recent5": x, y = x[-5:], y[-5:]
    if mode == "intensity3": return [1., float(np.mean(y[-3:] - x[-3:]))]
    return [float(x @ y / max(x @ x, 1e-12)), 0.]


def _capacity_rule(k, mode):
    increments = np.diff(k)
    if mode == "increment3": return float(max(0., increments[-3:].mean())), 0
    if mode == "increment5": return float(max(0., increments[-5:].mean())), 0
    return float(max(0., np.mean(increments[-3:] / k[-4:-1]))), 1


def calibrate(data, train_end=2020):
    a, g, k, service, shares, direct = _observations(data, train_end)
    if len(a) < 7: raise ValueError("Seven observed years required for training-only rolling selection")
    records = []
    demand_modes = ("full", "recent5", "intensity3")
    scores = {m: [] for m in demand_modes}
    for n in range(6, len(a)):
        for h in (1, 3):
            if n + h - 1 >= len(a): continue
            ratio = a.gdp_real_index.iloc[n+h-1] / a.gdp_real_index.iloc[n-1]
            for mode in demand_modes:
                beta, drift = _growth_fit(a.gdp_real_index.to_numpy()[:n], service[:n], mode)
                pred = service[n-1] * ratio ** beta * np.exp(drift*h)
                ape = abs(pred/service[n+h-1]-1)*100
                scores[mode].append(ape)
                records.append({"target":"useful_service_twh","method":mode,"origin_year":int(a.year.iloc[n-1]),"horizon":h,"observed":float(service[n+h-1]),"predicted":float(pred),"ape_percent":float(ape)})
    selected = min(scores, key=lambda m: np.mean(scores[m]))
    beta, demand_drift = _growth_fit(a.gdp_real_index.to_numpy(), service, selected)
    orders, exponents, cap_records = [], [], []
    for j, tech in enumerate(TECHS):
        modes = ("increment3", "increment5", "growth3")
        errors = {m: [] for m in modes}
        for n in range(6, len(a)):
            for h in (1, 3):
                if n+h-1 >= len(a): continue
                for mode in errors:
                    amount, exponent = _capacity_rule(k[:n, j], mode)
                    pred = k[n-1, j] * (1+amount)**h if exponent else k[n-1,j]+h*amount
                    ape = abs(pred/k[n+h-1,j]-1)*100
                    errors[mode].append(ape)
                    cap_records.append({"target":tech+"_capacity_gw","method":mode,"origin_year":int(a.year.iloc[n-1]),"horizon":h,"observed":float(k[n+h-1,j]),"predicted":float(pred),"ape_percent":float(ape)})
        mode = min(errors, key=lambda m: np.mean(errors[m]))
        amount, exponent = _capacity_rule(k[:,j], mode)
        orders.append(amount); exponents.append(exponent)
    logratios = np.log(shares / shares[:,[3]])
    drift = np.polyfit(a.year.to_numpy()-a.year.iloc[0], logratios, 1)[0]
    production = data["production"].loc[data["production"].year <= train_end].sort_values("year")
    production_growth = []
    production_diagnostics = []
    for fuel in FUELS:
        values = production[fuel+"_production_mtce_derived"].to_numpy(float)
        production_growth.append(float(np.mean(np.diff(np.log(values))[-3:])))
        consumption = production[fuel+"_consumption_mtce_derived"].to_numpy(float)
        x = consumption[:-1]/values[:-1]-1; y=np.diff(values)/values[:-1]
        fit = np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
        residual=y-np.column_stack([np.ones(len(x)),x])@fit
        production_diagnostics.append({"fuel":fuel,"feedback_slope":float(fit[1]),"r_squared":float(1-residual@residual/max(np.sum((y-y.mean())**2),1e-12)),"interpretation":"weak observational fit; slope not used as causal elasticity"})
    factors=[]
    for fuel in FUELS:
        factors.append(float((a[fuel+"_co2_mt"]/(a.energy_total_mtce*a[fuel+"_share"])).tail(3).median()))
    from src.v3_annual_calibrated import fit_expectations
    expectations=fit_expectations(data,train_end)
    return {"train_end":int(a.year.max()),"train_start":int(a.year.min()),"expectations":expectations,"service_growth_method":selected,"service_elasticity":beta,"service_drift":demand_drift,
            "share_logratio_drift":drift.tolist(),"capacity_rule_amount":orders,"capacity_rule_exponent":exponents,
            "training_reference_capacity":k[-1].tolist(),"utilisation":np.median(g[-3:]/(k[-3:]*8.76),axis=0).tolist(),
            "production_growth":production_growth,"production_diagnostics":production_diagnostics,"emission_factors":factors,
            "training_cv_records":records+cap_records,"demand_cv_mape":{m:float(np.mean(v)) for m,v in scores.items()},
            "scope":"Training-only low-dimensional rules; engineering cost/retirement/construction assumptions are not identified from these data"}


def _scenario(data, scenario):
    defaults=data["config"]["scenarios"]
    result=dict(defaults[scenario]) if isinstance(scenario,str) else {**defaults.get(scenario.get("name","baseline"),defaults["baseline"]),**scenario}
    for key in ("annual_budget_multiplier","supply_multiplier","import_multiplier","technology_cost_multiplier","carbon_price_multiplier","demand_growth_multiplier","reserve_multiplier"):
        value=float(result.get(key,1))
        if not np.isfinite(value) or value<0:raise ValueError(key+" must be finite and nonnegative")
    delay=float(result.get("construction_delay_multiplier",1))
    if not np.isfinite(delay) or delay<1 or delay>3:raise ValueError("construction_delay_multiplier must be within 1–3")
    return result


def initialize_state(data, params, start_year=2024):
    a,g,k,service,shares,direct=_observations(data,start_year)
    if int(a.year.iloc[-1]) != start_year: raise ValueError("Initial year requires complete observed data")
    prod=data["production"].loc[data["production"].year==start_year].iloc[0]
    x=np.zeros(NSTATE); x[0]=service[-1];x[1:5]=shares[-1];x[5:12]=k[-1]
    scale=np.exp((start_year-2024)*.10)
    x[12:14]=[73.76*scale,58.69*scale]
    x[14:17]=[prod[f+"_production_mtce_derived"] for f in FUELS]
    fuel=np.array([a.iloc[-1].energy_total_mtce*a.iloc[-1][f+"_share"] for f in FUELS])
    x[213:216]=np.maximum(fuel-x[14:17],0)
    x[17:20]=fuel*data["config"]["engineering"]["reserve_days"]/365
    x[204:211]=params["utilisation"];x[212]=100;x[216]=450
    biomass_migration=1.
    if start_year==2024:
        bio=data["supplement"].loc[data["supplement"].carrier=="biomass"].set_index("metric")
        biomass_migration=float(bio.loc["capacity_gw","value"])/k[-1,6]
        x[11]=float(bio.loc["capacity_gw","value"])
        x[210]=float(bio.loc["generation_twh","value"])/(x[11]*8.76)
    e=data["config"]["engineering"]
    # Historical capacity differences are NET additions. Reconstruct gross
    # orders by adding assumed replacement, rather than deducting retirement
    # twice. Retirement and the initial pipeline remain engineering assumptions.
    baseorder=np.array(params["capacity_rule_amount"])*np.where(np.array(params["capacity_rule_exponent"])==1,x[5:12],1)
    baseorder+=x[5:12]*np.array(e["retirement_fraction"][:7])
    supply_order=np.maximum(x[14:17]*(np.expm1(params["production_growth"])+np.array(e["retirement_fraction"][9:])),0)
    allorder=np.r_[baseorder,[max(.02*x[12],1),max(.015*x[13],1)],supply_order]
    # The unobserved initial pipeline is an explicit steady-throughput estimate,
    # not an observed construction register. Sensitivity can clear this queue.
    queue=x[20:QEND].reshape(12,QDEPTH)
    for j,L in enumerate(e["construction_years"]): queue[j,:int(L)]=allorder[j]
    net=np.maximum(np.diff(k[-4:],axis=0).mean(axis=0),0)
    net[6]*=biomass_migration
    reference=x[5:12].copy()
    for j,L in enumerate(e["construction_years"][:7]):
        for slot in range(int(L)):
            h=slot+1
            queue[j,slot]=reference[j]+net[j]*h-(1-e["retirement_fraction"][j])*(reference[j]+net[j]*(h-1))
    observed_generation=g[-1].copy()
    if start_year==2024:observed_generation[6]=float(bio.loc["generation_twh","value"])
    x[204:211]=observed_generation/(x[5:12]*8.76)
    nf_generation=float(g[-1,2:].sum())
    if start_year==2024: nf_generation+=float(bio.loc["generation_twh","value"])-g[-1,6]
    nf_factor=float(a.iloc[-1].energy_total_mtce*a.iloc[-1].nonfossil_share / nf_generation)
    runtime={**params,"engineering":e,"base_year":start_year,"initial_shares":shares[-1].tolist(),
             "initial_carbon_price":100.,"initial_service":float(service[-1]),"base_orders":allorder.tolist(),
             "initial_electricity":float(a.iloc[-1].electricity_demand_twh),"reference_capacity0":reference.tolist(),"reference_net_increment":net.tolist(),
             "base_fuel":fuel.tolist(),"nf_primary_factor":nf_factor,
             "base_budget":float(allorder@np.array(e["capital_cost_million_cny"])*1.15),
             "base_gdp":float(a.iloc[-1].gdp_real_index),"base_industry":float(a.iloc[-1].cement_co2_mt+a.iloc[-1].other_industry_co2_mt),
             "base_national_co2":float(a.iloc[-1].energy_co2_mt+a.iloc[-1].cement_co2_mt+a.iloc[-1].other_industry_co2_mt),
             "initial_biomass_generation_twh":float(bio.loc["generation_twh","value"]) if start_year==2024 else float(g[-1,6]),
             "biomass_initial_capacity_migration_ratio":biomass_migration,
             "gdp_history":{str(int(r.year)):float(r.gdp_real_index) for _,r in data["annual"].iterrows()}}
    runtime["gdp_years"]=data["annual"].year.to_numpy(float).tolist();runtime["gdp_values"]=data["annual"].gdp_real_index.to_numpy(float).tolist()
    return x,runtime


def step_numeric(state, p, scenario, year, gdp_ratio):
    """One annual causal transition. All orders enter a future queue slot."""
    s=scenario;e=p["engineering"];x=np.asarray(state,float).copy();n=x.copy()
    elapsed=year-p["base_year"]
    from src.v3_annual_calibrated import expectation_packet,forecast_level,reference_gross_orders
    service,forecast_electricity,base_share=expectation_packet(p,s,year)
    _,_,previous_base_share=expectation_packet(p,s,year-1)
    price=x[212]*s.get("carbon_price_multiplier",1)
    change=np.r_[price*np.array(p["emission_factors"])/(2000.),0]
    change-=np.r_[100*np.array(p["emission_factors"])/2000.,0]
    expected_decline=.015 if p["base_year"]==2024 else 0.
    expected_carbon=100+(3 if p["base_year"]==2024 else 0)*max(elapsed-1,0)
    change[:3]=(price-expected_carbon)*np.array(p["emission_factors"])/2000
    change[3]=.6*(s.get("technology_cost_multiplier",1)*np.exp(-(s.get("cost_decline_per_year",.015)-expected_decline)*elapsed)-1)
    logits=np.log(base_share)
    if s.get("cost_response",True): logits-=e["cost_response"]*change
    logits-=e["shortage_response"]*x[200:204]
    target=np.exp(logits-logits.max());target/=target.sum()
    adjusted_previous=base_share*x[1:5]/np.maximum(previous_base_share,1e-12);adjusted_previous/=adjusted_previous.sum()
    shares=(1-e["share_partial_adjustment"])*adjusted_previous+e["share_partial_adjustment"]*target
    direct_request=service*shares[:3]/np.array(e["direct_use_efficiency"][:3])/TWH_PER_MTCE
    demand=service*shares[3]/e["direct_use_efficiency"][3]
    queue=x[20:QEND].reshape(12,QDEPTH);commissioned=queue[:,0].copy()
    oldassets=np.r_[x[5:14],x[14:17]]
    retired=oldassets*np.array(e["retirement_fraction"])
    assets=np.maximum(oldassets-retired+commissioned,0)
    capacities=assets[:9];production_capacity=assets[9:]
    q=np.column_stack([queue[:,1:],np.zeros(12)])
    shock=year in s.get("annual_shock_years",[])
    util_reference=np.array(p["utilisation"],float)
    if p["base_year"]==2024:util_reference[6]=x[210] if elapsed==1 else p["engineering"].get("biomass_utilisation_nea2024",208.3/(45.99*8.76))
    potential=capacities[:7]*util_reference*8.76
    biomass_fuel_budget=p["initial_biomass_generation_twh"]*np.exp(s.get("biomass_feedstock_growth_per_year",.02)*elapsed)
    potential[6]=min(potential[6],biomass_fuel_budget)
    if shock: potential[3:6]*=1-s.get("renewable_shock_fraction",0)
    # Annual flexibility is a declared reduced-form acceptance envelope. It does
    # not claim to reproduce hourly curtailment or create energy from storage.
    vre_limit=min(.95,e["annual_vre_acceptance_base"]+e["annual_vre_acceptance_storage_gain"]*(capacities[7]+capacities[8])/max(demand/5,1))
    accepted_vre=min(potential[4:6].sum(),demand*vre_limit)
    nf=potential.copy();nf[:2]=0
    if potential[4:6].sum()>0: nf[4:6]*=accepted_vre/potential[4:6].sum()
    if nf.sum()>demand: nf*=demand/nf.sum()
    thermal_request=np.zeros(2);remaining=max(demand-nf.sum(),0)
    operating=np.array([300.,600.])+price*np.array([.82,.37])
    for j in np.argsort(operating):
        thermal_request[j]=min(remaining,potential[j]);remaining-=thermal_request[j]
    requested=direct_request+np.array([thermal_request[0]/(.4*TWH_PER_MTCE),0,thermal_request[1]/(.55*TWH_PER_MTCE)])
    domestic_available=production_capacity*s.get("supply_multiplier",1)
    import_capacity=x[213:216]*max(gdp_ratio,1)
    imports=import_capacity*s.get("import_multiplier",1)*(1-s.get("import_shock_fraction",.5) if shock else 1)
    ordinary=domestic_available+imports
    release=np.minimum(x[17:20]*e["max_stock_release_fraction_per_year"],np.maximum(requested-ordinary,0))
    delivered=np.minimum(requested,ordinary+release)
    ratio=np.minimum(delivered/np.maximum(requested,1e-12),1)
    direct_delivered=direct_request*ratio
    generation=nf.copy();generation[0]=thermal_request[0]*ratio[0];generation[1]=thermal_request[1]*ratio[2]
    powerfuel=np.array([generation[0]/(.4*TWH_PER_MTCE),0,generation[1]/(.55*TWH_PER_MTCE)])
    stock_target=requested*e["reserve_days"]*s.get("reserve_multiplier",1)/365
    recharge=np.minimum(np.maximum(stock_target-(x[17:20]-release),0),requested*e["max_recharge_fraction_of_demand"])
    recharge=np.minimum(recharge,np.maximum(ordinary-requested,0))
    if shock: recharge*=0
    used=direct_delivered+powerfuel
    inflow=used+recharge-release
    actual_imports=np.minimum(imports,inflow);domestic=inflow-actual_imports
    stocks=x[17:20]+domestic+actual_imports-used
    stocks[np.abs(stocks)<1e-10]=0
    unserved=max(demand-generation.sum(),0);curtail=float((potential[2:]-generation[2:]).sum())
    shortage=np.r_[1-ratio,unserved/max(demand,1e-12)]
    utilisation=generation/np.maximum(capacities[:7]*8.76,1e-12)
    gross=float(used@np.array(p["emission_factors"]))
    nf_primary=float(generation[2:].sum()*p["nf_primary_factor"])
    primary=float(used.sum()+nf_primary)
    industry=p["base_industry"]*(service/p["initial_service"])*np.exp(-.01*elapsed)
    total=gross+industry;net=total-e["sink_mt"]
    # Investors share one finite budget. Forecast scarcity, utilisation/revenue
    # and curtailment affect orders; no terminal-year asset target exists.
    costs=np.array(e["capital_cost_million_cny"],float)
    techlevel=s.get("technology_cost_multiplier",1)*np.exp(-s.get("cost_decline_per_year",.015)*elapsed)
    costs[4:8]*=techlevel
    base=np.array(p["base_orders"],float)
    delays=np.ceil(np.array(e["construction_years"])*s.get("construction_delay_multiplier",1)).astype(int)
    base[:7]=reference_gross_orders(p,delays,elapsed)
    base[9:]=np.maximum(production_capacity*(np.expm1(p["production_growth"])+np.array(e["retirement_fraction"][9:])),0)
    relative=np.ones(12)
    market_price=1200. if unserved>1e-6 else (float(operating[generation[:2]>1e-6].max()) if (generation[:2]>1e-6).any() else 300.)
    variable_cost=np.array([300.,600.,20.,15.,5.,5.,250.])
    carbon_per_twh=np.array([p["emission_factors"][0]/(.4*TWH_PER_MTCE),p["emission_factors"][2]/(.55*TWH_PER_MTCE),0,0,0,0,0])
    revenue=generation*market_price
    operating_expense=generation*(variable_cost+price*carbon_per_twh)
    annual_capital=costs[:7]*capacities[:7]*.08
    profit=revenue-operating_expense-annual_capital
    reference_cost=np.array(e["capital_cost_million_cny"][:7],float)
    reference_cost[4:6]*=np.exp(-expected_decline*elapsed)
    reference_profit=generation*(market_price-variable_cost-expected_carbon*carbon_per_twh)-reference_cost*capacities[:7]*.08
    profit_ratio=(profit-reference_profit)/np.maximum(reference_cost*capacities[:7]*.08,1)
    if s.get("cost_response",True):
        relative[:7]=np.clip(1+.25*profit_ratio,0,2)
    feedback=np.zeros(12)
    if s.get("investment_feedback",True):
        # A direct oil/gas shortage cannot justify building arbitrary electricity
        # assets unless users actually switch service requests to electricity.
        stress=unserved/max(demand,1)+x[203]
        previous_electricity=x[0]*x[4]/e["direct_use_efficiency"][3]
        expected_one_step=forecast_level(previous_electricity,gdp_ratio,1,p["expectations"]["electricity_fit"])
        previous_generation=x[204:211]*x[5:12]*8.76
        expected_generation=previous_generation*expected_one_step/max(previous_electricity,1e-12)
        innovation=(generation-expected_generation)/np.maximum(capacities[:7]*8.76,1e-12)
        feedback[:7]=p["expectations"]["utilisation_response"]*capacities[:7]*innovation+e["investment_feedback"]*capacities[:7]*stress
        feedback[4:6]-=e["investment_feedback"]*capacities[4:6]*curtail/max(potential[4:6].sum(),1)
        feedback[7]=e["investment_feedback"]*(curtail+unserved)/max(4*.365,1)/20
        feedback[8]=feedback[7]*.15
        feedback[9:]=e["investment_feedback"]*production_capacity*(1-ratio)
    wanted=np.maximum(base+feedback,0)*relative
    biomass_future_budget=p["initial_biomass_generation_twh"]*np.exp(s.get("biomass_feedstock_growth_per_year",.02)*(elapsed+e["construction_years"][6]))
    biomass_uncommitted=max(0,biomass_future_budget/(util_reference[6]*8.76)-capacities[6]-q[6].sum())
    wanted[6]=min(wanted[6],biomass_uncommitted)
    limit=np.maximum(oldassets*np.array([.15,.20,.15,.08,.40,.50,.20,.50,.15,.15,.10,.20]),base*1.5)
    wanted=np.minimum(wanted,limit)
    budget=p["base_budget"]*(service/p["initial_service"])**.7*s.get("annual_budget_multiplier",1)
    spend=float(wanted@costs);orders=wanted*min(1,budget/max(spend,1e-12));spend=float(orders@costs)
    delayfactor=s.get("construction_delay_multiplier",1)
    delays=np.ceil(np.array(e["construction_years"])*delayfactor).astype(int)
    if (delays<1).any() or (delays>QDEPTH).any():raise ValueError("Construction delay outside supported 1–15 years")
    for j,L in enumerate(delays): q[j,L-1]+=orders[j]
    n[0]=service;n[1:5]=shares;n[5:14]=capacities;n[14:17]=production_capacity;n[17:20]=stocks;n[20:QEND]=q.ravel();n[200:204]=shortage;n[204:211]=utilisation;n[211]=curtail
    reference=p["base_national_co2"]*np.exp(-s.get("carbon_reference_decline_per_year",.02)*max(year-2030,0))
    carbon_feedback=e["carbon_feedback_gain_cny_t"]*(total/reference-1) if s.get("carbon_feedback",True) else 0
    n[212]=np.clip(x[212]+s.get("carbon_price_drift_cny_t",3)+carbon_feedback,0,e["carbon_price_max_cny_t"])
    n[213:216]=import_capacity;n[216]=market_price
    metrics=np.r_[demand,generation.sum(),unserved,primary,gross,net,max(net,0),curtail,spend,budget,nf_primary,total,price,service,generation,direct_request,direct_delivered,powerfuel,domestic,actual_imports,stocks,stocks-x[17:20],orders,commissioned,retired,feedback,costs,profit,revenue,operating_expense,annual_capital,market_price,reference]
    return n,metrics


def _row(x,m,scenario,year):
    names=("electricity_demand_twh","electricity_generation_twh","electricity_unserved_twh","energy_total_mtce","energy_co2_mt","net_co2_mt","required_removal_mt","curtailment_twh","investment_spend_million_cny","investment_budget_million_cny","nonfossil_primary_mtce","gross_national_co2_mt","carbon_price_cny_t","useful_service_demand_twh")
    row={"scenario":scenario,"year":year,**dict(zip(names,m[:14])),**{f+"_capacity_gw":x[5+j] for j,f in enumerate((*TECHS,"battery","pumped"))}}
    row["construction_pending_gw"]=float(x[20:QEND].reshape(12,QDEPTH)[:9].sum())
    for j,t in enumerate(ASSETS):
        row[t+"_pending"]=float(x[20:QEND].reshape(12,QDEPTH)[j].sum())
        suffix="_unit_cost_million_cny_per_gw" if j<9 else "_unit_cost_million_cny_per_mtce_per_year"
        row[t+suffix]=m[90+j]
    for j,t in enumerate(TECHS):row[t+"_generation_twh"]=m[14+j]
    for z,name in enumerate(("direct_demand","direct_delivered","power_use","domestic_supply","imports","stock","stock_change")):
        for j,f in enumerate(FUELS):row[f+"_"+name+"_mtce"]=m[21+z*3+j]
    row["fuel_balance_error_mtce"]=max(abs(row[f+"_domestic_supply_mtce"]+row[f+"_imports_mtce"]-row[f+"_stock_change_mtce"]-row[f+"_direct_delivered_mtce"]-row[f+"_power_use_mtce"]) for f in FUELS)
    row["electricity_balance_error_twh"]=abs(row["electricity_generation_twh"]+row["electricity_unserved_twh"]-row["electricity_demand_twh"])
    row["nonfossil_share"]=row["nonfossil_primary_mtce"]/row["energy_total_mtce"]
    actions=[]
    for j,agent in enumerate(ASSETS):
        actions.append({"year":year,"scenario":scenario,"agent":agent,"order":m[42+j],"commissioned":m[54+j],"retired":m[66+j],"feedback_signal":m[78+j],"capital_spend_million_cny":m[42+j]*m[90+j],"simulated_profit_million_cny":float(m[102+j]) if j<7 else np.nan,"simulated_revenue_million_cny":float(m[109+j]) if j<7 else np.nan,"simulated_operating_cost_million_cny":float(m[116+j]) if j<7 else np.nan,"annual_capital_charge_million_cny":float(m[123+j]) if j<7 else np.nan,"unit":"GW" if j<9 else "Mtce/year"})
    row["market_price_cny_mwh"]=m[130];row["policy_reference_co2_mt"]=m[131]
    for j,f in enumerate((*FUELS,"electricity")):row["user_"+f+"_service_share"]=float(x[1+j])
    return row,actions


def simulate(data,params,scenario,end_year=2060,start_year=2024):
    s=_scenario(data,scenario);x,p=initialize_state(data,params,start_year);x[212]=s.get("carbon_price_start_cny_t",100)
    if s.get("clear_initial_pipeline",False):x[20:QEND]=0
    rows=[];actions=[];gdp=p["base_gdp"]
    for year in range(start_year+1,end_year+1):
        if str(year) in p["gdp_history"]:nextgdp=p["gdp_history"][str(year)]
        else: nextgdp=gdp*(1+s["gdp_growth"][0 if year<=2030 else 1 if year<=2040 else 2])
        x,m=step_numeric(x,p,s,year,nextgdp/gdp);gdp=nextgdp
        row,acts=_row(x,m,s["name"],year)
        row["biomass_generation_budget_twh_per_year"]=p["initial_biomass_generation_twh"]*np.exp(s.get("biomass_feedstock_growth_per_year",.02)*(year-start_year))
        rows.append(row);actions.extend(acts)
    return pd.DataFrame(rows),pd.DataFrame(actions)


def historical_validation(data,params):
    records=[]
    for origin in range(2017,2024):
        p=calibrate(data,origin)
        observed=data["annual"].set_index("year")
        for h in (1,3):
            if origin+h>2024:continue
            out,_=simulate(data,p,{"name":"baseline","cost_decline_per_year":0,"carbon_feedback":False,"carbon_price_drift_cny_t":0},origin+h,origin)
            a,g,k,svc,shares,direct=_observations(data,origin)
            future=data["power"].loc[data["power"].year==origin+h].set_index("technology")
            for tech in TECHS:
                actual=float(future.loc[tech,"capacity_gw"]);recent=float(np.diff(k[-4:,TECHS.index(tech)]).mean());base=k[-1,TECHS.index(tech)]
                predictions={"dynamic_abm":float(out.iloc[-1][tech+"_capacity_gw"]),"persistence":base,"recent_increment3":base+h*recent}
                for method,pred in predictions.items():records.append({"origin_year":origin,"horizon":h,"target_year":origin+h,"target":tech+"_capacity_gw","method":method,"observed":actual,"predicted":pred,"ape_percent":abs(pred/actual-1)*100})
            actual=float(observed.loc[origin+h,"electricity_demand_twh"]);base=float(observed.loc[origin,"electricity_demand_twh"]);slope=float(np.diff(a.electricity_demand_twh.tail(4)).mean())
            for method,pred in {"dynamic_abm":float(out.iloc[-1].electricity_demand_twh),"persistence":base,"recent_increment3":base+h*slope}.items():records.append({"origin_year":origin,"horizon":h,"target_year":origin+h,"target":"electricity_demand_twh","method":method,"observed":actual,"predicted":pred,"ape_percent":abs(pred/actual-1)*100})
    return pd.DataFrame(records)


def run(root):
    root=Path(root);out=root/"outputs/v3";out.mkdir(parents=True,exist_ok=True)
    data=load_data(root);params=calibrate(data);snapshots={};frames=[];actionframes=[]
    for name in data["config"]["scenarios"]:
        frame,actions=simulate(data,params,name);frames.append(frame);actionframes.append(actions)
        for year in (2030,2060):
            r=frame.loc[frame.year==year].iloc[0];caps={t:float(r[t+"_capacity_gw"]) for t in TECHS}
            caps.update(battery_power=float(r.battery_capacity_gw),battery_energy=float(r.battery_capacity_gw)*4,pumped_power=float(r.pumped_capacity_gw),pumped_energy=float(r.pumped_capacity_gw)*6)
            snapshots[name+"_"+str(year)]={"year":year,"scenario":name,"electricity_demand_twh":float(r.electricity_demand_twh),"capacity_gw":caps,"fuel_domestic_supply_mtce_per_year":{f:float(r[f+"_domestic_supply_mtce"]) for f in FUELS},"fuel_import_supply_mtce_per_year":{f:float(r[f+"_imports_mtce"]) for f in FUELS},"fuel_initial_stocks_mtce":{f:float(r[f+"_stock_mtce"]) for f in FUELS},"direct_fuel_demand_mtce_per_year":{f:float(r[f+"_direct_demand_mtce"]) for f in FUELS},"fuel_efficiency":{"coal":.4,"gas":.55},"carbon_price_cny_t":float(r.carbon_price_cny_t),"biomass_generation_budget_twh_per_year":float(r.biomass_generation_budget_twh_per_year)}
    pd.concat(frames).to_csv(out/"annual_scenarios.csv",index=False);pd.concat(actionframes).to_csv(out/"agent_actions.csv",index=False)
    cv=params.pop("training_cv_records");pd.DataFrame(cv).to_csv(out/"calibration_training_cv.csv",index=False)
    (out/"calibration.json").write_text(json.dumps(params,indent=2,ensure_ascii=False),"utf-8")
    (out/"snapshots.json").write_text(json.dumps(snapshots,indent=2),"utf-8")
    hist=historical_validation(data,params);hist.to_csv(out/"historical_rolling.csv",index=False)
    hist.groupby(["target","method","horizon"]).ape_percent.agg(["mean","count"]).reset_index().to_csv(out/"historical_metrics.csv",index=False)
    holdout,acts=simulate(data,params,{"name":"baseline","cost_decline_per_year":0,"carbon_feedback":False,"carbon_price_drift_cny_t":0},2024,2020)
    holdout["validation_type"]="development_rollout_condition_given_observed_GDP"
    holdout.to_csv(out/"historical_conditioned_gdp_2021_2024.csv",index=False)
    assumptions={"engineering":data["config"]["engineering"],"calibrated":params,"boundaries":["Four aggregate useful-service groups, not observed industrial/transport sectors","Domestic production-consumption gaps initialise import channels; these are not observed net imports","Unobserved initial construction pipeline estimated from recent throughput","No CCS deployment is imposed; required removals report the remaining gap","Annual VRE acceptance is an engineering envelope, not calibrated hourly curtailment","2024 biomass uses paired NEA capacity/generation; prior Ember bioenergy is a different taxonomy","2021-2024 is development validation, not a new blind test","Construction/retirement/cost preferences are engineering parameters; annual data cannot identify them separately"]}
    (out/"assumptions.json").write_text(json.dumps(assumptions,indent=2,ensure_ascii=False),"utf-8")
    x,p=initialize_state(data,params,2024)
    savemat(out/"annual_native_inputs.mat",{"initial_state":x,"parameters":p,"scenarios":data["config"]["scenarios"]},long_field_names=True)
    print("V3 annual complete",len(pd.concat(frames)),"rows;",len(pd.concat(actionframes)),"actions;",len(hist),"rolling comparisons")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);run(parser.parse_args().root)
