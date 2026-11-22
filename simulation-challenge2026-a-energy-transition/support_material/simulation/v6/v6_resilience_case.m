function c=v6_resilience_case(snapshot,supplyPolicy,hours,eventMode)
% State-continuous designed events; independent MATLAB protocol implementation.
% AI-assisted, OpenAI Codex, 2026-10-18. Fixed dates do not scale with duration.
arguments
    snapshot = []
    supplyPolicy (1,1) string = "inherited"
    hours (1,1) double {mustBeInteger,mustBeGreaterThan(hours,1)} = 2160
    eventMode (1,1) string = "repeated"
end
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
addpath(fullfile(root,'simulation','v3'));
p=jsondecode(fileread(fullfile(root,'config','v6_resilience.json')));
assert(any(supplyPolicy==["inherited","contract"]) && any(eventMode==["single","repeated"]),'Unknown event/supply policy.');
assert(hours<=p.hours,'Duration must crop the fixed design, not extend or rescale it.');
if isempty(snapshot)
    file=fullfile(root,'outputs','v6','resilience','native_inputs.json');
    assert(isfile(file),'V6:Inputs','Run the V6 fixed Python cohort preparation first.');
    input=jsondecode(fileread(file));snapshot=input.snapshots.(input.default_snapshot_key);
end
c=v3_hourly_case(snapshot,"normal",hours);c.snapshot=snapshot;
t=(0:hours-1)';c.stress_mask=false(hours,1);
phase=repmat("recovery_2",hours,1);phase(t<p.events(2,1))="recovery_1";phase(t<p.warmup_hours)="warmup";
extra=zeros(hours,3);normalImports=zeros(1,3);
for j=1:3,normalImports(j)=snapshot.fuel_import_capacity_mtce_per_year.(c.fuels(j))/8760;end
eventCount=2;if eventMode=="single",eventCount=1;end
for event=1:eventCount
    lo=p.events(event,1);hi=p.events(event,2);mask=t>=lo & t<hi;
    c.stress_mask=c.stress_mask|mask;phase(mask)="shock_"+event;
    c.fuel_import_arrivals_mtce(mask,:)=c.fuel_import_arrivals_mtce(mask,:).*p.import_retention(:)';
    for tech=["wind","solar","hydro"]
        j=find(c.generators==tech);c.available_gw(mask,j)=c.available_gw(mask,j)*p.renewable_retention.(tech);
    end
    c.demand_gw(mask)=c.demand_gw(mask)*p.event_demand_multiplier;
    if supplyPolicy=="contract"
        first=hi+p.contract_lead_hours;
        if event<2,last=p.events(event+1,1);else,last=p.hours;end
        rate=min(normalImports*p.contract_max_fraction_normal_import_rate, ...
            c.initial_fuel_stocks_mtce*p.contract_max_fraction_initial_stock_per_event/(last-first));
        delivery=t>=first & t<last;extra(delivery,:)=extra(delivery,:)+rate;
    end
end
if eventMode=="single",phase(t>=p.events(1,2))="recovery_1";end
c.fuel_arrivals_mtce=c.fuel_domestic_arrivals_mtce+c.fuel_import_arrivals_mtce+extra;
baseValue=sum(extra,1)*c.config.fuel_value_million_cny_mtce';
c.phase_label=phase;c.emergency_arrivals_mtce=extra;c.event_protocol=p;
c.supply_policy=supplyPolicy;c.event_mode=eventMode;
c.contract_base_value_million_cny=baseValue;
c.contract_premium_million_cny=baseValue*p.contract_premium_fraction;
c.contract_purchase_value_million_cny=baseValue*(1+p.contract_premium_fraction);
c.shock="two_event";if eventMode=="single",c.shock="single_event";end
c.information_contract="COMMON_24H_PERFECT_FORECAST_ROLLING_FIRST_ACTION";
c.boundary=string(p.scope);
end
