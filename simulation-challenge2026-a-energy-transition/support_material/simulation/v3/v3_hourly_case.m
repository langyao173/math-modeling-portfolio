function c = v3_hourly_case(snapshot,shock,hours)
% Raw engineered experiment inputs, not a replay of Python control actions.
arguments
    snapshot (1,1) struct
    shock (1,1) string = "normal"
    hours (1,1) double {mustBeInteger,mustBeGreaterThan(hours,1)} = 720
end
assert(any(shock==["normal","import_cut","renewable_drought","compound"]),'Unknown shock.');
c=struct; c.year=snapshot.year; c.scenario=string(snapshot.scenario);
c.hours=hours; c.shock=shock; c.dt_h=1;
c.generators=["coal","gas","nuclear","hydro","wind","solar","biomass"];
c.fuels=["coal","oil","gas"]; c.stores=["battery","pumped"];
c.capacity_gw=snapshot.capacity_gw;
t=(0:hours-1)';
daily=1+.12*cos(2*pi*(mod(t,24)-15)/24)+.03*cos(4*pi*(mod(t,24)-19)/24);
c.demand_gw=snapshot.electricity_demand_twh*1000/8760*daily;
wind=max(0,min(.85,.29*(1+.35*sin(2*pi*t/72)+.12*sin(2*pi*t/17))));
solar=max(0,sin(pi*(mod(t,24)-6)/12))*.17/mean(max(0,sin(pi*((0:23)-6)/12)));
factors=[.90*ones(hours,1),.92*ones(hours,1),.88*ones(hours,1),.35*ones(hours,1),wind,solar,.65*ones(hours,1)];
c.available_gw=zeros(hours,7);
for j=1:7
    c.available_gw(:,j)=factors(:,j)*snapshot.capacity_gw.(c.generators(j));
end
if snapshot.capacity_gw.biomass>0
    assert(isfield(snapshot,'biomass_generation_budget_twh_per_year'), ...
        'Nonzero biomass capacity requires annual feedstock generation budget.');
    c.biomass_generation_budget_twh_per_year=snapshot.biomass_generation_budget_twh_per_year;
else
    c.biomass_generation_budget_twh_per_year=0;
end
assert(isfinite(c.biomass_generation_budget_twh_per_year) && c.biomass_generation_budget_twh_per_year>=0);
c.available_gw(:,7)=min(c.available_gw(:,7),c.biomass_generation_budget_twh_per_year*1000/8760);
c.fuel_domestic_arrivals_mtce=zeros(hours,3);
c.fuel_import_arrivals_mtce=zeros(hours,3);
c.direct_fuel_demand_mtce=zeros(hours,3); c.initial_fuel_stocks_mtce=zeros(1,3);
for j=1:3
    f=c.fuels(j);
    c.fuel_domestic_arrivals_mtce(:,j)=snapshot.fuel_domestic_supply_mtce_per_year.(f)/8760;
    c.fuel_import_arrivals_mtce(:,j)=snapshot.fuel_import_supply_mtce_per_year.(f)/8760;
    c.direct_fuel_demand_mtce(:,j)=snapshot.direct_fuel_demand_mtce_per_year.(f)/8760;
    c.initial_fuel_stocks_mtce(j)=snapshot.fuel_initial_stocks_mtce.(f);
end
assert(all(c.initial_fuel_stocks_mtce>=0),'Initial fuel stocks must be nonnegative.');
start=min(120,floor(hours/4)); stop=min(600,max(floor(hours/4)+1,floor(5*hours/6)));
c.stress_mask=t>=start & t<stop;
if any(shock==["import_cut","compound"])
    c.fuel_import_arrivals_mtce(c.stress_mask,:)=c.fuel_import_arrivals_mtce(c.stress_mask,:).*[.50,.05,.05];
end
if any(shock==["renewable_drought","compound"])
    c.available_gw(c.stress_mask,[5,6,4])=c.available_gw(c.stress_mask,[5,6,4]).*[.25,.55,.70];
end
if shock=="compound", c.demand_gw(c.stress_mask)=c.demand_gw(c.stress_mask)*1.08; end
c.fuel_arrivals_mtce=c.fuel_domestic_arrivals_mtce+c.fuel_import_arrivals_mtce;
c.fuel_efficiency=snapshot.fuel_efficiency;
c.initial_thermal_gw=[0,0]; c.initial_debt_gwh=0;
for j=1:2
    name=c.stores(j); eta=sqrt([.90,.78]);
    c.storage.(name)=struct('power_gw',snapshot.capacity_gw.(name+"_power"), ...
        'energy_gwh',snapshot.capacity_gw.(name+"_energy"), ...
        'initial_soc_gwh',.5*snapshot.capacity_gw.(name+"_energy"), ...
        'eta_charge',eta(j),'eta_discharge',eta(j));
end
c.config=struct('forecast_hours',24,'deadline_hours',24, ...
    'ramp_fraction',[.15,.70],'flex_fraction',.08,'repay_fraction',.20, ...
    'generation_cost_cny_kwh',[.32,.65,.09,.025,0,0,.35], ...
    'carbon_price_cny_t',snapshot.carbon_price_cny_t, ...
    'unserved_cost_cny_kwh',30,'direct_shortage_cost_cny_kwh_th',30, ...
    'storage_cycle_cost_cny_kwh',.005,'shift_cost_cny_kwh',.003);
c.config.fuel_value_million_cny_mtce=[.32*8141*c.fuel_efficiency.coal,3500,.65*8141*c.fuel_efficiency.gas];
c.config.stored_electricity_value_million_cny_gwh=.32;
c.config.stock_holding_rate_per_year=.05;
c.evidence_class="DESIGNED_HOURLY_DYNAMIC_EXPERIMENT";
c.information_contract="COMMON_24H_PERFECT_FORECAST_ROLLING_FIRST_ACTION";
c.boundary="Aggregate bus, continuous ramp constraints, no integer commitment/network; hydro available-flow input; engineered hourly profiles; native MATLAB independently optimizes from Simulink feedback states.";
end
