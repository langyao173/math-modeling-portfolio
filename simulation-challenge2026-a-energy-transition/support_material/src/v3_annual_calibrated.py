"""Calibrated expectation helpers used by the single annual stock-flow kernel.

Forecasts supply beliefs and proposed orders, never realised capacity, supply,
generation or consumption. All realised stocks remain in v3_annual.step_numeric.
"""
from __future__ import annotations
import numpy as np
from src.v3_calibration_candidate import calibrate_candidate


def fit_expectations(data,train_end):
    fitted=calibrate_candidate(data,train_end)
    return {"service_fit":fitted["service_fit"],"electricity_fit":fitted["electricity_fit"],
            "utilisation_response":fitted["utilisation_response"],"utilisation_response_raw":fitted["utilisation_response_raw"],
            "utilisation_selected":fitted["utilisation_selected"],"utilisation_cv_mape":fitted["utilisation_cv_mape"],
            "service_cv":fitted["service_cv"],"electricity_cv":fitted["electricity_cv"],
            "n_utilisation_response_pairs":fitted["n_utilisation_response_pairs"],"train_end":int(train_end)}


def forecast_level(initial,gdp_ratio,elapsed,fit,growth_multiplier=1.):
    if fit["mode"]=="increment3":return max(initial+fit["increment"]*elapsed*growth_multiplier,1e-9)
    return float(initial*gdp_ratio**(fit["beta"]*growth_multiplier)*np.exp(fit["drift"]*elapsed))


def gdp_level(p,s,year):
    if str(year) in p["gdp_history"]:return p["gdp_history"][str(year)]
    value=p["base_gdp"]
    for t in range(p["base_year"]+1,year+1):
        if str(t) in p["gdp_history"]:value=p["gdp_history"][str(t)]
        else:value*=1+s["gdp_growth"][0 if t<=2030 else 1 if t<=2040 else 2]
    return value


def expectation_packet(p,s,year):
    h=year-p["base_year"];current_gdp=gdp_level(p,s,year)/p["base_gdp"]
    mult=s.get("demand_growth_multiplier",1.)
    efficiency=np.exp(-s.get("efficiency_improvement_per_year",0)*h)
    service=forecast_level(p["initial_service"],current_gdp,h,p["expectations"]["service_fit"],mult)*efficiency
    electricity=forecast_level(p["initial_electricity"],current_gdp,h,p["expectations"]["electricity_fit"],mult)*efficiency
    electric_share=np.clip(electricity*.9/service,1e-8,.95)
    fossil=np.log(np.array(p["initial_shares"][:3]))+np.array(p["share_logratio_drift"][:3])*h
    fossil=np.exp(fossil-fossil.max());fossil=fossil/fossil.sum()*(1-electric_share)
    share=np.r_[fossil,electric_share]
    return service,electricity,share


def reference_gross_orders(p,delays,elapsed):
    initial=np.array(p["reference_capacity0"]);net=np.array(p["reference_net_increment"])
    retirement=np.array(p["engineering"]["retirement_fraction"][:7])
    future=initial+net*(elapsed+np.array(delays[:7]))
    previous=initial+net*(elapsed+np.array(delays[:7])-1)
    return np.maximum(future-(1-retirement)*previous,0)
