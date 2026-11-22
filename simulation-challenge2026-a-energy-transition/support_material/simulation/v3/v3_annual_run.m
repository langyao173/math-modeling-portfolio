function result=v3_annual_run(scenarioName,factors)
% Execute the autonomous native Simulink transition from the observed 2024 state.
if nargin<1, scenarioName='baseline'; end
if nargin<2, factors=struct(); end
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder)); addpath(folder);
input=asdouble(load(fullfile(root,'outputs','v3','annual_native_inputs.mat')));
if ~isfield(input.scenarios,scenarioName), error('V3:Scenario','Unknown scenario'); end
scenario=input.scenarios.(scenarioName); fields=fieldnames(factors);
for j=1:numel(fields), scenario.(fields{j})=factors.(fields{j}); end
global V3_ANNUAL_CONTEXT
V3_ANNUAL_CONTEXT=struct('parameters',input.parameters,'scenario',scenario,'initial_state',input.initial_state(:));
V3_ANNUAL_CONTEXT.initial_state(213)=scenario.carbon_price_start_cny_t;
if isfield(scenario,'clear_initial_pipeline') && scenario.clear_initial_pipeline, V3_ANNUAL_CONTEXT.initial_state(21:200)=0; end
modelPath=v3_build_annual_model(); [~,name]=fileparts(modelPath);
assignin('base','V3_ANNUAL_CONTEXT',V3_ANNUAL_CONTEXT);
pace=0; if isfield(factors,'pace_rate'), pace=factors.pace_rate; end
simInput=Simulink.SimulationInput(name);
if pace>0, simInput=simInput.setModelParameter('EnablePacing','on','PacingRate',num2str(pace)); else, simInput=simInput.setModelParameter('EnablePacing','off'); end
if pace>0,open_system([name '/Native annual state dynamics']);end
simulationOutput=v3_execute(simInput,'annual',pace);
logged=simulationOutput.get('annual_native_trace'); values=squeeze(logged.signals.values);
if size(values,2)~=349, values=values'; end
state=values(:,1:217); m=values(:,218:349); year=input.parameters.base_year+1+round(logged.time);
names={'electricity_demand_twh','electricity_generation_twh','electricity_unserved_twh','energy_total_mtce','energy_co2_mt','net_co2_mt','required_removal_mt','curtailment_twh','investment_spend_million_cny','investment_budget_million_cny','nonfossil_primary_mtce','gross_national_co2_mt','carbon_price_cny_t','useful_service_demand_twh'};
trace=array2table(m(:,1:14),'VariableNames',names); trace=addvars(trace,year,'Before',1); trace.scenario=repmat(string(scenarioName),height(trace),1);
tech={'coal','gas','nuclear','hydro','wind','solar','biomass','battery','pumped'};
assets=[tech,{'fuel_coal','fuel_oil','fuel_gas'}]; fuels={'coal','oil','gas'};
trace.construction_pending_gw=sum(state(:,21:155),2);
for j=1:9, trace.([tech{j} '_capacity_gw'])=state(:,5+j); end
for j=1:7, trace.([tech{j} '_generation_twh'])=m(:,14+j); end
labels={'direct_demand','direct_delivered','power_use','domestic_supply','imports','stock','stock_change'};
for z=1:7, for j=1:3, trace.([fuels{j} '_' labels{z} '_mtce'])=m(:,21+(z-1)*3+j); end, end
trace.fuel_balance_error_mtce=zeros(height(trace),1);
for j=1:3
    f=fuels{j}; residual=trace.([f '_domestic_supply_mtce'])+trace.([f '_imports_mtce'])-trace.([f '_stock_change_mtce'])-trace.([f '_direct_delivered_mtce'])-trace.([f '_power_use_mtce']);
    trace.fuel_balance_error_mtce=max(trace.fuel_balance_error_mtce,abs(residual));
end
trace.electricity_balance_error_twh=abs(trace.electricity_generation_twh+trace.electricity_unserved_twh-trace.electricity_demand_twh);
trace.nonfossil_share=trace.nonfossil_primary_mtce./trace.energy_total_mtce;
for j=1:12
    trace.([assets{j} '_pending'])=sum(state(:,21+(j-1)*15:35+(j-1)*15),2);
    if j<10,suffix='_unit_cost_million_cny_per_gw';else,suffix='_unit_cost_million_cny_per_mtce_per_year';end
    trace.([assets{j} suffix])=m(:,90+j);
end
rows=height(trace)*12;
actions=table(repelem(year,12),repmat(string(assets)',height(trace),1),reshape(m(:,43:54)',rows,1),reshape(m(:,55:66)',rows,1),reshape(m(:,67:78)',rows,1),reshape(m(:,79:90)',rows,1),reshape((m(:,43:54).*m(:,91:102))',rows,1),'VariableNames',{'year','agent','order','commissioned','retired','feedback_signal','capital_spend_million_cny'});
trace.market_price_cny_mwh=m(:,131);trace.policy_reference_co2_mt=m(:,132);
biomassGrowth=.02;if isfield(scenario,'biomass_feedstock_growth_per_year'),biomassGrowth=scenario.biomass_feedstock_growth_per_year;end
trace.biomass_generation_budget_twh_per_year=input.parameters.initial_biomass_generation_twh*exp(biomassGrowth*(year-input.parameters.base_year));
serviceGroups={'coal','oil','gas','electricity'};
for j=1:4,trace.(['user_' serviceGroups{j} '_service_share'])=state(:,1+j);end
actions.simulated_profit_million_cny=nan(rows,1);actions.simulated_revenue_million_cny=nan(rows,1);actions.simulated_operating_cost_million_cny=nan(rows,1);actions.annual_capital_charge_million_cny=nan(rows,1);
for j=1:7
    positions=j:12:rows;
    actions.simulated_profit_million_cny(positions)=m(:,102+j);
    actions.simulated_revenue_million_cny(positions)=m(:,109+j);
    actions.simulated_operating_cost_million_cny(positions)=m(:,116+j);
    actions.annual_capital_charge_million_cny(positions)=m(:,123+j);
end
snapshots=struct();
for targetYear=[2030 2060]
    r=trace(trace.year==targetYear,:); capacity=struct();
    for j=1:7, capacity.(tech{j})=r.([tech{j} '_capacity_gw']); end
    capacity.battery_power=r.battery_capacity_gw; capacity.battery_energy=r.battery_capacity_gw*4;
    capacity.pumped_power=r.pumped_capacity_gw; capacity.pumped_energy=r.pumped_capacity_gw*6;
    snapshot=struct('year',targetYear,'scenario',scenarioName,'electricity_demand_twh',r.electricity_demand_twh,'capacity_gw',capacity,'fuel_efficiency',struct('coal',.4,'gas',.55),'carbon_price_cny_t',r.carbon_price_cny_t,'biomass_generation_budget_twh_per_year',r.biomass_generation_budget_twh_per_year);
    for j=1:3
        f=fuels{j}; snapshot.fuel_domestic_supply_mtce_per_year.(f)=r.([f '_domestic_supply_mtce']);
        snapshot.fuel_import_supply_mtce_per_year.(f)=r.([f '_imports_mtce']); snapshot.fuel_initial_stocks_mtce.(f)=r.([f '_stock_mtce']); snapshot.direct_fuel_demand_mtce_per_year.(f)=r.([f '_direct_demand_mtce']);
    end
    snapshots.([scenarioName '_' num2str(targetYear)])=snapshot;
end
[peak,index]=max(trace.energy_co2_mt);
peakYear=trace.year(index);initialObserved=input.parameters.base_national_co2-input.parameters.base_industry;
atInitial=initialObserved>=peak;
if atInitial,peak=initialObserved;peakYear=input.parameters.base_year;end
metrics=struct('peak_year',peakYear,'peak_mt',peak,'peak_at_observed_initial_year',atInitial,'peak_at_horizon_endpoint',peakYear==trace.year(end),'net_2060_mt',trace.net_co2_mt(end),'required_removal_2060_mt',trace.required_removal_mt(end),'max_balance_error',max([trace.fuel_balance_error_mtce;trace.electricity_balance_error_twh]),'source_mode','NATIVE_SIMULINK_CLOSED_LOOP','capture_capacity_imposed_mt',0);
result=struct('trace',trace,'actions',actions,'metrics',metrics,'snapshots',snapshots,'state',state,'raw_metrics',m,'simulation_output',simulationOutput,'model_path',modelPath);
end

function value=asdouble(value)
if isstruct(value)
    fields=fieldnames(value);
    for i=1:numel(value), for j=1:numel(fields), value(i).(fields{j})=asdouble(value(i).(fields{j})); end, end
elseif isnumeric(value)
    value=double(value);
end
end
