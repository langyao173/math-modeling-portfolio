% V6 import-channel correction, based on simulation/v3/v3_annual_run.m, SHA256 669207364be09604b4ba68fae1e4ec0e13e1128f7a970625fa38e64abaf79e66.
% Original V3 files remain frozen. V6 interventions are executed in this kernel.
function result=v6_annual_run(scenarioName,factors)
% Execute the autonomous native Simulink transition from the observed 2024 state.
if nargin<1, scenarioName='baseline'; end
if nargin<2, factors=struct(); end
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder)); addpath(folder);addpath(fullfile(root,'simulation','v3'),'-end');
input=asdouble(load(fullfile(root,'outputs','v6','native','annual_native_inputs.mat')));
if ~isfield(input.cases,scenarioName), error('V6:Scenario','Unknown V6 native case'); end
selected=input.cases.(scenarioName);scenario=selected.scenario;rules=selected.mechanisms;
fields=fieldnames(factors);
for j=1:numel(fields)
    key=fields{j};if isfield(rules,key),rules.(key)=factors.(key);else,scenario.(key)=factors.(key);end
end
if isfield(scenario,'investment_priority'),assert(numel(scenario.investment_priority)==12 && all(isfinite(scenario.investment_priority)) && all(scenario.investment_priority>=0),'V6:Priorities','Twelve finite nonnegative priorities required');end
if isfield(scenario,'budget_multipliers_by_period'),assert(numel(scenario.budget_multipliers_by_period)==3 && all(isfinite(scenario.budget_multipliers_by_period)) && all(scenario.budget_multipliers_by_period>=0),'V6:Budgets','Three finite nonnegative period budgets required');end
global V6_ANNUAL_CONTEXT
V6_ANNUAL_CONTEXT=struct('parameters',input.parameters,'scenario',scenario,'mechanisms',rules,'initial_state',input.initial_state(:));
V6_ANNUAL_CONTEXT.initial_state(213)=scenario.carbon_price_start_cny_t;
if strcmp(rules.coal_import_channel,'zero_channel'),V6_ANNUAL_CONTEXT.initial_state(214)=0;end
if strcmp(rules.coal_import_channel,'zero_matched_supply')
    V6_ANNUAL_CONTEXT.initial_state(214)=0;
    V6_ANNUAL_CONTEXT.parameters.domestic_retention_fraction(1)=input.parameters.base_fuel(1)/input.initial_state(15);
end
if isfield(scenario,'clear_initial_pipeline') && scenario.clear_initial_pipeline, V6_ANNUAL_CONTEXT.initial_state(21:200)=0; end
modelPath=v6_build_annual_model(); [~,name]=fileparts(modelPath);
assignin('base','V6_ANNUAL_CONTEXT',V6_ANNUAL_CONTEXT);
pace=0; if isfield(factors,'pace_rate'), pace=factors.pace_rate; end
simInput=Simulink.SimulationInput(name);
if pace>0, simInput=simInput.setModelParameter('EnablePacing','on','PacingRate',num2str(pace)); else, simInput=simInput.setModelParameter('EnablePacing','off'); end
if pace>0,open_system([name '/Native annual state dynamics']);end
simulationOutput=v3_execute(simInput,'annual',pace);
logged=simulationOutput.get('annual_native_trace'); values=squeeze(logged.signals.values);
if size(values,2)~=349, values=values'; end
state=values(:,1:217); m=values(:,218:349); year=input.parameters.base_year+1+round(logged.time);
names={'electricity_demand_twh','electricity_generation_twh','electricity_unserved_twh','energy_total_mtce','energy_co2_mt','net_co2_mt','required_removal_mt','curtailment_twh','investment_spend_million_cny','investment_budget_million_cny','nonfossil_primary_mtce','gross_national_co2_mt','carbon_price_cny_t','useful_service_demand_twh'};
trace=array2table(m(:,1:14),'VariableNames',names); trace=addvars(trace,year,'Before',1); trace.scenario=repmat(string(scenario.name),height(trace),1);trace.case_id=repmat(string(scenarioName),height(trace),1);
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
for j=1:3
    trace.([fuels{j} '_import_capacity_mtce_per_year'])=state(:,213+j);
    trace.([fuels{j} '_production_capacity_mtce_per_year'])=state(:,14+j);
    trace.([fuels{j} '_domestic_retention_fraction'])=repmat(V6_ANNUAL_CONTEXT.parameters.domestic_retention_fraction(j),height(trace),1);
end
rates=scenario.gdp_growth(:);gdp=input.parameters.base_gdp;gdpValues=zeros(height(trace),1);
for i=1:height(trace)
    y=trace.year(i);if y<=2030,k=1;elseif y<=2040,k=2;else,k=3;end
    gdp=gdp*(1+rates(k));gdpValues(i)=gdp;
end
trace.gdp_real_index=gdpValues;
trace.energy_intensity_index_base_year_100=100*trace.energy_total_mtce/input.parameters.base_primary_energy_mtce./(gdpValues/input.parameters.base_gdp);
trace.carbon_intensity_index_base_year_100=100*trace.gross_national_co2_mt/input.parameters.base_national_co2./(gdpValues/input.parameters.base_gdp);
trace.fixed_sink_coverage_rate=input.parameters.engineering.sink_mt./max(trace.gross_national_co2_mt,1e-12);
trace.carbon_neutralization_rate=min(trace.fixed_sink_coverage_rate,1);
trace.carbon_neutral_target_met=trace.net_co2_mt<=0;

for j=1:12
    trace.([assets{j} '_pending'])=sum(state(:,21+(j-1)*15:35+(j-1)*15),2);
    if j<10,suffix='_unit_cost_million_cny_per_gw';else,suffix='_unit_cost_million_cny_per_mtce_per_year';end
    trace.([assets{j} suffix])=m(:,90+j);
end
rows=height(trace)*12;
actions=table(repelem(year,12),repmat(string(assets)',height(trace),1),reshape(m(:,43:54)',rows,1),reshape(m(:,55:66)',rows,1),reshape(m(:,67:78)',rows,1),reshape(m(:,79:90)',rows,1),reshape((m(:,43:54).*m(:,91:102))',rows,1),'VariableNames',{'year','agent','order','commissioned','retired','feedback_signal','capital_spend_million_cny'});
trace.policy_reference_co2_mt=m(:,132);
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
trace.market_price_cny_mwh=m(:,131);
% Reconstruct the investment benchmark and signal from completed native values.
% The state/output vector stays 217/132 wide; these are an explanatory ledger.
for field={'reference_profit_million_cny','reference_capital_charge_million_cny','normalized_profit_signal','profit_order_multiplier','market_price_cny_mwh'}
    actions.(field{1})=zeros(rows,1);
end
actions.profit_reference_rule=repmat(string(rules.investor_profit_rule),rows,1);
actions.profit_reference_origin_year=repmat(input.parameters.investment_reference.origin_year,rows,1);
p=input.parameters;anchor=p.investment_reference;variableCost=[300;600;20;15;5;5;250];
carbonPerTwh=[p.emission_factors(1)/(.4*(29.3076/3.6));p.emission_factors(3)/(.55*(29.3076/3.6));0;0;0;0;0];
initialCost=anchor.capital_cost_million_cny_per_gw(:);
referencePerGw=8.76*anchor.utilisation(:).*(anchor.market_price_cny_mwh-variableCost-anchor.carbon_price_cny_t*carbonPerTwh)-.08*initialCost;
for j=1:7
    positions=j:12:rows;capacity=trace.([tech{j} '_capacity_gw']);
    reference=capacity*referencePerGw(j);normalizer=max(.08*initialCost(j)*capacity,1);
    if strcmp(rules.investor_profit_rule,'legacy_v4')
        expectedDecline=0;expectedCarbon=100+zeros(height(trace),1);
        if p.base_year==2024,expectedDecline=.015;expectedCarbon=100+3*max(year-p.base_year-1,0);end
        referenceCost=initialCost(j)+zeros(height(trace),1);
        if j==5 || j==6,referenceCost=referenceCost.*exp(-expectedDecline*(year-p.base_year));end
        reference=trace.([tech{j} '_generation_twh']).*(trace.market_price_cny_mwh-variableCost(j)-expectedCarbon*carbonPerTwh(j))-referenceCost.*capacity*.08;
        normalizer=max(referenceCost.*capacity*.08,1);
    end
    signal=(actions.simulated_profit_million_cny(positions)-reference)./normalizer;
    multiplier=ones(height(trace),1);
    if scenario.cost_response && rules.investor_profit_response,multiplier=min(max(1+.25*signal,0),2);end
    actions.reference_profit_million_cny(positions)=reference;actions.reference_capital_charge_million_cny(positions)=normalizer;
    actions.normalized_profit_signal(positions)=signal;actions.profit_order_multiplier(positions)=multiplier;
    actions.market_price_cny_mwh(positions)=trace.market_price_cny_mwh;
end
actions.scenario=repmat(string(scenario.name),rows,1);actions.case_id=repmat(string(scenarioName),rows,1);
actions.unit=repmat([repmat("GW",9,1);repmat("Mtce/year",3,1)],height(trace),1);
snapshots=struct();
for targetYear=[2030 2060]
    r=trace(trace.year==targetYear,:); capacity=struct();
    for j=1:7, capacity.(tech{j})=r.([tech{j} '_capacity_gw']); end
    capacity.battery_power=r.battery_capacity_gw; capacity.battery_energy=r.battery_capacity_gw*4;
    capacity.pumped_power=r.pumped_capacity_gw; capacity.pumped_energy=r.pumped_capacity_gw*6;
    snapshot=struct('year',targetYear,'scenario',scenarioName,'electricity_demand_twh',r.electricity_demand_twh,'capacity_gw',capacity,'fuel_efficiency',struct('coal',.4,'gas',.55),'carbon_price_cny_t',r.carbon_price_cny_t,'biomass_generation_budget_twh_per_year',r.biomass_generation_budget_twh_per_year);
    for j=1:3
        f=fuels{j}; snapshot.fuel_domestic_supply_mtce_per_year.(f)=r.([f '_domestic_supply_mtce']);
        snapshot.fuel_import_supply_mtce_per_year.(f)=r.([f '_imports_mtce']); snapshot.fuel_import_capacity_mtce_per_year.(f)=r.([f '_import_capacity_mtce_per_year']); snapshot.fuel_initial_stocks_mtce.(f)=r.([f '_stock_mtce']); snapshot.direct_fuel_demand_mtce_per_year.(f)=r.([f '_direct_demand_mtce']);
    end
    snapshots.([scenarioName '_' num2str(targetYear)])=snapshot;
end
[peak,index]=max(trace.energy_co2_mt);
peakYear=trace.year(index);initialObserved=input.parameters.base_national_co2-input.parameters.base_industry;
atInitial=initialObserved>=peak;
if atInitial,peak=initialObserved;peakYear=input.parameters.base_year;end
metrics=struct('peak_year',peakYear,'peak_mt',peak,'peak_at_observed_initial_year',atInitial,'peak_at_horizon_endpoint',peakYear==trace.year(end),'net_2060_mt',trace.net_co2_mt(end),'required_removal_2060_mt',trace.required_removal_mt(end),'max_balance_error',max([trace.fuel_balance_error_mtce;trace.electricity_balance_error_twh]),'source_mode','V6_NATIVE_SIMULINK_CLOSED_LOOP','capture_capacity_imposed_mt',0);
executedInputs=struct('case_name',scenarioName,'scenario',scenario,'mechanisms',rules, ...
    'native_input_file','outputs/v6/native/annual_native_inputs.mat', ...
    'native_input_sha256',shaFile(fullfile(root,'outputs','v6','native','annual_native_inputs.mat')), ...
    'scope','Fixed initial observations and calibrated parameters plus resolved policies/mechanisms; no future Python state or control path');
result=struct('trace',trace,'actions',actions,'metrics',metrics,'snapshots',snapshots,'state',state,'raw_metrics',m,'simulation_output',simulationOutput,'model_path',modelPath,'executed_inputs',executedInputs);
end

function digest=shaFile(path)
fid=fopen(path,'rb');assert(fid~=-1,'V6:Hash','Cannot read executed native input');
bytes=fread(fid,Inf,'*uint8');fclose(fid);md=java.security.MessageDigest.getInstance('SHA-256');md.update(bytes);
digest=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end

function value=asdouble(value)
if isstruct(value)
    fields=fieldnames(value);
    for i=1:numel(value), for j=1:numel(fields), value(i).(fields{j})=asdouble(value(i).(fields{j})); end, end
elseif isnumeric(value)
    value=double(value);
end
end
