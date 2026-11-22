"""V6 gross-import channels with an explicitly reconciled domestic supply map.

Static V5 kernel with independent gross channels, a disclosed origin-year
domestic retention map and scarcity-price interventions. V5 files stay frozen.
AI-assisted: OpenAI Codex, 2026-10-18.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from src import v3_annual as frozen

TECHS=frozen.TECHS
ASSETS=frozen.ASSETS
FUELS=frozen.FUELS
TWH_PER_MTCE=frozen.TWH_PER_MTCE
NSTATE=frozen.NSTATE
QDEPTH=frozen.QDEPTH
QEND=frozen.QEND
def load_data(root):
    data=frozen.load_data(root)
    channels=pd.read_csv(Path(root)/"data/v6_fuel_import_channels.csv")
    required={"year","fuel","gross_imports_mtce","source_id","conversion_basis"}
    if not required.issubset(channels.columns):
        raise ValueError("Gross-import table is missing required fields")
    if channels.duplicated(["year","fuel"]).any() or not channels.fuel.isin(FUELS).all():
        raise ValueError("Gross-import table requires unique year/fuel rows")
    values=channels.gross_imports_mtce.to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError("Gross import quantities must be finite and nonnegative")
    data["import_channels"]=channels
    return data


calibrate=frozen.calibrate


DEFAULT_MECHANISMS={
    "consumer_choice":"responsive",
    "carbon_feedback":True,
    "investor_profit_response":True,
    "investor_utilisation_response":True,
    "construction_years":"native",
    "investor_profit_rule":"origin_anchored",
    "coal_import_channel":"gross_trade",
    "market_price_rule":"step",
}


def initialize_state(data, params, start_year=2024, *, mechanisms=None):
    """Freeze the investment benchmark at the simulation origin.

    Capacity factors are reconstructed from origin-year generation/capacity.
    The 450 CNY/MWh market state and 100 CNY/t carbon state are engineering
    initial conditions, not observed tariffs. No future realised price enters.
    """
    state,p=frozen.initialize_state(data,params,start_year)
    rules=resolve_mechanisms(mechanisms)
    channels=data["import_channels"].loc[data["import_channels"].year==start_year].set_index("fuel")
    if set(channels.index)!=set(FUELS):
        raise ValueError("Independent gross imports required for every fuel at the initial year")
    observed_imports=channels.loc[list(FUELS),"gross_imports_mtce"].to_numpy(float)
    consumption=np.asarray(p["base_fuel"],float)
    production=state[14:17].copy()
    retained=consumption-observed_imports
    if (retained<0).any() or (production<=0).any():
        raise ValueError("Gross-import conversion cannot be reconciled with observed fuel use")
    retention=retained/production
    if rules["coal_import_channel"]=="zero_matched_supply":
        observed_imports=observed_imports.copy();observed_imports[0]=0.
        retention=retention.copy();retention[0]=consumption[0]/production[0]
    if (retention>1+1e-12).any():
        raise ValueError("Gross-import conversion requires domestic retention above physical production: " + str(dict(zip(FUELS,retention.tolist()))))
    state[213:216]=observed_imports
    if rules["coal_import_channel"]=="zero_channel":state[213]=0.
    p["domestic_retention_fraction"]=retention.tolist()
    initial_row=data["annual"].loc[data["annual"].year==start_year].iloc[0]
    p["base_primary_energy_mtce"]=float(initial_row.energy_total_mtce)
    p["initial_fuel_account"]=[{
        "year":int(start_year),"fuel":fuel,"observed_production_mtce":float(production[j]),
        "observed_consumption_mtce":float(consumption[j]),
        "gross_import_capacity_mtce":float(state[213+j]),
        "domestic_retention_fraction":float(retention[j]),
        "model_domestic_retained_mtce":float(production[j]*retention[j]),
        "unallocated_balance_adjustment_mtce":float(production[j]+state[213+j]-consumption[j]),
        "model_initial_supply_mtce":float(production[j]*retention[j]+state[213+j]),
        "initial_supply_minus_observed_use_mtce":float(production[j]*retention[j]+state[213+j]-consumption[j]),
        "source_id":str(channels.loc[fuel,"source_id"]),
        "conversion_basis":str(channels.loc[fuel,"conversion_basis"]),
        "coal_channel_rule":rules["coal_import_channel"],
    } for j,fuel in enumerate(FUELS)]
    p["investment_reference"]={
        "origin_year":int(start_year),
        "market_price_cny_mwh":float(state[216]),
        "carbon_price_cny_t":float(state[212]),
        "utilisation":state[204:211].tolist(),
        "capital_cost_million_cny_per_gw":list(p["engineering"]["capital_cost_million_cny"][:7]),
    }
    return state,p


def investor_return_terms(generation, capacities, costs, market_price, carbon_price, p):
    """Annual money terms (million CNY), and dimensionless excess return.

    A TWh times CNY/MWh is million CNY. Reference generation/GW is 8.76*u0
    TWh/GW/year. Normalization uses the initial 8% annual capital charge on
    current installed capacity, so changing fleet scale cannot change the ROI.
    """
    generation=np.asarray(generation,float);capacities=np.asarray(capacities,float)
    costs=np.asarray(costs,float);anchor=p["investment_reference"]
    variable=np.array([300.,600.,20.,15.,5.,5.,250.])
    carbon=np.array([p["emission_factors"][0]/(.4*TWH_PER_MTCE),
                     p["emission_factors"][2]/(.55*TWH_PER_MTCE),0.,0.,0.,0.,0.])
    initial_cost=np.asarray(anchor["capital_cost_million_cny_per_gw"],float)
    reference_per_gw=(8.76*np.asarray(anchor["utilisation"],float)
        *(anchor["market_price_cny_mwh"]-variable-anchor["carbon_price_cny_t"]*carbon)
        -.08*initial_cost)
    reference=capacities*reference_per_gw
    profit=generation*(market_price-variable-carbon_price*carbon)-.08*costs*capacities
    normalizer=np.maximum(.08*initial_cost*capacities,1.)
    return {"reference_profit_million_cny":reference,"profit_million_cny":profit,
            "reference_capital_charge_million_cny":normalizer,
            "signal":(profit-reference)/normalizer}


def profit_order_multiplier(signal, enabled):
    relative=np.ones(12)
    if enabled:relative[:7]=np.clip(1+.25*np.asarray(signal,float),0,2)
    return relative


def attach_investor_ledger(frame,actions,p,scenario,rules):
    """Expose the actual price/return rule without widening the numeric kernel."""
    result=actions.copy()
    for column in ("reference_profit_million_cny","reference_capital_charge_million_cny",
                   "normalized_profit_signal","profit_order_multiplier","market_price_cny_mwh"):
        result[column]=0.
    result["profit_reference_rule"]=rules["investor_profit_rule"]
    result["profit_reference_origin_year"]=p["investment_reference"]["origin_year"]
    for _,row in frame.iterrows():
        cap=np.array([row[t+"_capacity_gw"] for t in TECHS])
        gen=np.array([row[t+"_generation_twh"] for t in TECHS])
        mask=(result.year==row.year)&result.agent.isin(TECHS)
        # Actions retain ASSETS order, matching generation/capacity vectors.
        selected=result.loc[mask]
        unit_cost=np.array([row[t+"_unit_cost_million_cny_per_gw"] for t in TECHS])
        terms=investor_return_terms(gen,cap,unit_cost,row.market_price_cny_mwh,row.carbon_price_cny_t,p)
        reference=terms["reference_profit_million_cny"];normalizer=terms["reference_capital_charge_million_cny"]
        if rules["investor_profit_rule"]=="legacy_v4":
            elapsed=int(row.year)-p["base_year"]
            refcost=np.array(p["engineering"]["capital_cost_million_cny"][:7],float)
            expected_decline=.015 if p["base_year"]==2024 else 0.
            refcost[4:6]*=np.exp(-expected_decline*elapsed)
            expected_carbon=100+(3 if p["base_year"]==2024 else 0)*max(elapsed-1,0)
            variable=np.array([300.,600.,20.,15.,5.,5.,250.])
            carbon=np.array([p["emission_factors"][0]/(.4*TWH_PER_MTCE),p["emission_factors"][2]/(.55*TWH_PER_MTCE),0,0,0,0,0])
            reference=gen*(row.market_price_cny_mwh-variable-expected_carbon*carbon)-refcost*cap*.08
            normalizer=np.maximum(refcost*cap*.08,1)
        signal=(selected.simulated_profit_million_cny.to_numpy()-reference)/normalizer
        result.loc[mask,"reference_profit_million_cny"]=reference
        result.loc[mask,"reference_capital_charge_million_cny"]=normalizer
        result.loc[mask,"normalized_profit_signal"]=signal
        result.loc[mask,"profit_order_multiplier"]=profit_order_multiplier(signal,scenario.get("cost_response",True) and rules["investor_profit_response"])[:7]
        result.loc[mask,"market_price_cny_mwh"]=row.market_price_cny_mwh
    return result


def resolve_mechanisms(mechanisms=None):
    supplied=mechanisms or {}
    if set(supplied)-set(DEFAULT_MECHANISMS):
        raise ValueError("Unknown mechanism names: "+str(sorted(set(supplied)-set(DEFAULT_MECHANISMS))))
    rules={**DEFAULT_MECHANISMS,**supplied}
    if rules["consumer_choice"] not in ("responsive","frozen_initial","exogenous_trend"):
        raise ValueError("Unknown consumer_choice mechanism")
    if rules["construction_years"] not in ("native","minimum_one_year"):
        raise ValueError("Construction must be native or minimum_one_year; zero lag is not implemented")
    if rules["investor_profit_rule"] not in ("origin_anchored", "legacy_v4"):
        raise ValueError("Unknown investor_profit_rule")
    for key in ("carbon_feedback","investor_profit_response","investor_utilisation_response"):
        if not isinstance(rules[key],bool):raise ValueError(key+" must be Boolean")
    if rules["coal_import_channel"] not in ("gross_trade","zero_channel","zero_matched_supply"):
        raise ValueError("Unknown coal import channel intervention")
    if rules["market_price_rule"] not in ("step","smooth"):
        raise ValueError("Unknown market price rule")
    return rules


def scarcity_market_price(marginal_cost, unserved, demand, tier, rule):
    """The legacy step and a continuous shortage-fraction response.

    The smooth intervention reaches half its nonnegative scarcity premium at
    a 1% annual electricity shortage; the 1% scale is fixed before execution.
    """
    if rule=="step":return float(tier if unserved>1e-6 else marginal_cost)
    fraction=unserved/max(demand,1e-12)
    return float(marginal_cost+max(tier-marginal_cost,0.)*fraction/(fraction+.01))


def _decision_scenario(data,scenario):
    s=frozen._scenario(data,scenario)
    for key,size in (("investment_priority",12),("budget_multipliers_by_period",3)):
        values=np.asarray(s.get(key,np.ones(size)),float)
        if values.shape!=(size,) or not np.isfinite(values).all() or (values<0).any():
            raise ValueError(key+" must contain "+str(size)+" finite nonnegative values")
    return s


def _engineering_data(data,overrides):
    if not overrides:return data
    engineering=data["config"]["engineering"]
    if set(overrides)-set(engineering):raise ValueError("Unknown engineering override")
    for key,value in overrides.items():
        original=np.asarray(engineering[key]); replacement=np.asarray(value,float)
        if replacement.shape!=original.shape or not np.isfinite(replacement).all() or (replacement<0).any():
            raise ValueError("Invalid engineering override: "+key)
    altered={**data,"config":{**data["config"],"engineering":{**engineering,**deepcopy(overrides)}}}
    return altered


def step_numeric(state, p, scenario, year, gdp_ratio, mechanisms=None):
    """One causal year; new orders always enter a future construction slot."""
    rules = {**DEFAULT_MECHANISMS, **(mechanisms or {})}
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
    if rules["consumer_choice"] == "frozen_initial":
        shares=np.array(p["initial_shares"],float)
    elif rules["consumer_choice"] == "exogenous_trend":
        shares=base_share.copy()
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
    domestic_available=production_capacity*np.asarray(p["domestic_retention_fraction"])*s.get("supply_multiplier",1)
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
    marginal_cost=float(operating[generation[:2]>1e-6].max()) if (generation[:2]>1e-6).any() else e["no_thermal_market_price_cny_mwh"]
    market_price=scarcity_market_price(marginal_cost,unserved,demand,e["market_scarcity_price_cny_mwh"],rules["market_price_rule"])
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
    if rules["investor_profit_rule"] == "origin_anchored":
        profit_ratio=investor_return_terms(generation,capacities[:7],costs[:7],market_price,price,p)["signal"]
    relative=profit_order_multiplier(profit_ratio,s.get("cost_response",True) and rules["investor_profit_response"])
    feedback=np.zeros(12)
    if s.get("investment_feedback",True) and rules["investor_utilisation_response"]:
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
    wanted*=np.asarray(s.get("investment_priority",np.ones(12)),float)
    # The minimum-lag contrast leaves the native investment expectation anchor
    # unchanged; only actual new-project commissioning and its physical feedstock
    # limit use the shorter lead. Subsequent realised feedback remains active.
    if rules["construction_years"] == "minimum_one_year": delays=np.ones(12,dtype=int)
    # Counterfactual construction must check feedstock at the actual new-project
    # commissioning year. Native-default delays equal the frozen V3 values.
    biomass_future_budget=p["initial_biomass_generation_twh"]*np.exp(s.get("biomass_feedstock_growth_per_year",.02)*(elapsed+delays[6]))
    biomass_uncommitted=max(0,biomass_future_budget/(util_reference[6]*8.76)-capacities[6]-q[6].sum())
    wanted[6]=min(wanted[6],biomass_uncommitted)
    limit=np.maximum(oldassets*np.array([.15,.20,.15,.08,.40,.50,.20,.50,.15,.15,.10,.20]),base*1.5)
    wanted=np.minimum(wanted,limit)
    period=0 if year<=2030 else 1 if year<=2040 else 2
    budget=p["base_budget"]*(service/p["initial_service"])**.7*s.get("annual_budget_multiplier",1)
    budget*=s.get("budget_multipliers_by_period",[1.,1.,1.])[period]
    spend=float(wanted@costs);orders=wanted*min(1,budget/max(spend,1e-12));spend=float(orders@costs)
    delayfactor=s.get("construction_delay_multiplier",1)
    delays=np.ceil(np.array(e["construction_years"])*delayfactor).astype(int)
    if rules["construction_years"] == "minimum_one_year": delays=np.ones(12,dtype=int)
    if (delays<1).any() or (delays>QDEPTH).any():raise ValueError("Construction delay outside supported 1–15 years")
    for j,L in enumerate(delays): q[j,L-1]+=orders[j]
    n[0]=service;n[1:5]=shares;n[5:14]=capacities;n[14:17]=production_capacity;n[17:20]=stocks;n[20:QEND]=q.ravel();n[200:204]=shortage;n[204:211]=utilisation;n[211]=curtail
    reference=p["base_national_co2"]*np.exp(-s.get("carbon_reference_decline_per_year",.02)*max(year-2030,0))
    carbon_feedback=e["carbon_feedback_gain_cny_t"]*(total/reference-1) if s.get("carbon_feedback",True) and rules["carbon_feedback"] else 0
    n[212]=np.clip(x[212]+s.get("carbon_price_drift_cny_t",3)+carbon_feedback,0,e["carbon_price_max_cny_t"])
    n[213:216]=import_capacity;n[216]=market_price
    metrics=np.r_[demand,generation.sum(),unserved,primary,gross,net,max(net,0),curtail,spend,budget,nf_primary,total,price,service,generation,direct_request,direct_delivered,powerfuel,domestic,actual_imports,stocks,stocks-x[17:20],orders,commissioned,retired,feedback,costs,profit,revenue,operating_expense,annual_capital,market_price,reference]
    return n,metrics


def simulate(data,params,scenario,end_year=2060,start_year=2024,*,mechanisms=None,engineering_overrides=None,investor_ledger=True):
    """Run an annual counterfactual, returning the V3-compatible frame/actions.

    Period multipliers (2025-30/2031-40/2041-60) multiply the scenario-wide
    budget multiplier. Twelve investment priorities multiply desired orders
    BEFORE the original annual build-rate, biomass and shared-budget caps.
    The minimum-one-year switch applies only to new projects; the observed-
    state inferred initial pipeline is unchanged.
    """
    case_data=_engineering_data(data,engineering_overrides)
    s=_decision_scenario(case_data,scenario);rules=resolve_mechanisms(mechanisms)
    x,p=initialize_state(case_data,params,start_year,mechanisms=rules);x[212]=s.get("carbon_price_start_cny_t",100)
    if s.get("clear_initial_pipeline",False):x[20:QEND]=0
    rows=[];actions=[];gdp=p["base_gdp"]
    for year in range(start_year+1,end_year+1):
        if str(year) in p["gdp_history"]:nextgdp=p["gdp_history"][str(year)]
        else:nextgdp=gdp*(1+s["gdp_growth"][0 if year<=2030 else 1 if year<=2040 else 2])
        x,m=step_numeric(x,p,s,year,nextgdp/gdp,rules);gdp=nextgdp
        row,acts=frozen._row(x,m,s["name"],year)
        row["gdp_real_index"]=gdp
        row["energy_intensity_index_base_year_100"]=100*row["energy_total_mtce"]/p["base_primary_energy_mtce"]/(gdp/p["base_gdp"])
        row["carbon_intensity_index_base_year_100"]=100*row["gross_national_co2_mt"]/p["base_national_co2"]/(gdp/p["base_gdp"])
        row["fixed_sink_coverage_rate"]=p["engineering"]["sink_mt"]/max(row["gross_national_co2_mt"],1e-12)
        row["carbon_neutralization_rate"]=min(p["engineering"]["sink_mt"]/max(row["gross_national_co2_mt"],1e-12),1.)
        row["carbon_neutral_target_met"]=bool(row["net_co2_mt"]<=0.)
        for j,fuel in enumerate(FUELS):
            row[fuel+"_import_capacity_mtce_per_year"]=float(x[213+j])
            row[fuel+"_domestic_retention_fraction"]=p["domestic_retention_fraction"][j]
            row[fuel+"_production_capacity_mtce_per_year"]=float(x[14+j])
        row["biomass_generation_budget_twh_per_year"]=p["initial_biomass_generation_twh"]*np.exp(s.get("biomass_feedstock_growth_per_year",.02)*(year-start_year))
        rows.append(row);actions.extend(acts)
    frame=pd.DataFrame(rows); actions=pd.DataFrame(actions)
    return frame,attach_investor_ledger(frame,actions,p,s,rules) if investor_ledger else actions


def snapshot_from_row(row,scenario_name=None):
    """Transfer a realised annual state to the unchanged hourly-case boundary."""
    caps={tech:float(row[tech+"_capacity_gw"]) for tech in TECHS}
    caps.update(battery_power=float(row["battery_capacity_gw"]),battery_energy=float(row["battery_capacity_gw"])*4,
                pumped_power=float(row["pumped_capacity_gw"]),pumped_energy=float(row["pumped_capacity_gw"])*6)
    return {"year":int(row["year"]),"scenario":scenario_name or str(row["scenario"]),
            "electricity_demand_twh":float(row["electricity_demand_twh"]),"capacity_gw":caps,
            "fuel_domestic_supply_mtce_per_year":{f:float(row[f+"_domestic_supply_mtce"]) for f in FUELS},
            "fuel_import_supply_mtce_per_year":{f:float(row[f+"_imports_mtce"]) for f in FUELS},
            "fuel_import_capacity_mtce_per_year":{f:float(row[f+"_import_capacity_mtce_per_year"]) for f in FUELS},
            "fuel_initial_stocks_mtce":{f:float(row[f+"_stock_mtce"]) for f in FUELS},
            "direct_fuel_demand_mtce_per_year":{f:float(row[f+"_direct_demand_mtce"]) for f in FUELS},
            "fuel_efficiency":{"coal":.4,"gas":.55},"carbon_price_cny_t":float(row["carbon_price_cny_t"]),
            "biomass_generation_budget_twh_per_year":float(row["biomass_generation_budget_twh_per_year"])}
