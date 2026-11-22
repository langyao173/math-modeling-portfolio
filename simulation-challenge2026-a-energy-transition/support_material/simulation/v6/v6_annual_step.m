% V6 import-channel correction, based on simulation/v3/v3_annual_step.m, SHA256 df3e7746638fa3baad4856c973690ca118fab8f1846e5b9427a7b5d9254a8923.
% Original V3 files remain frozen. V6 interventions are executed in this kernel.
function out = v6_annual_step(u)
% Autonomous annual stock-flow decisions; Unit Delay owns all dynamic state.
% This function consumes only prior state, clock, fixed calibration and scenario.
global V6_ANNUAL_CONTEXT
p=V6_ANNUAL_CONTEXT.parameters; s=V6_ANNUAL_CONTEXT.scenario; e=p.engineering;
rules=V6_ANNUAL_CONTEXT.mechanisms;
x=u(1:217); x=x(:); n=x; year=p.base_year+1+round(u(218)); elapsed=year-p.base_year;
if year<=2030, gdpRate=s.gdp_growth(1); elseif year<=2040, gdpRate=s.gdp_growth(2); else, gdpRate=s.gdp_growth(3); end
gdpRatio=1+gdpRate;
% Consumer agent: beliefs, carrier choices and requests precede physical clearing.
[service,forecastElectricity,baseShare]=v3_annual_expectations(p,s,year);
[~,~,previousBaseShare]=v3_annual_expectations(p,s,year-1);
price=x(213)*getv(s,'carbon_price_multiplier',1);
expectedDecline=0;expectedCarbon=100;
if p.base_year==2024,expectedDecline=.015;expectedCarbon=100+3*max(elapsed-1,0);end
change=[(price-expectedCarbon)*p.emission_factors(:)/2000;0];
change(4)=.6*(getv(s,'technology_cost_multiplier',1)*exp(-(getv(s,'cost_decline_per_year',.015)-expectedDecline)*elapsed)-1);
logits=log(baseShare);
if getv(s,'cost_response',true), logits=logits-e.cost_response*change; end
logits=logits-e.shortage_response*x(201:204);
target=exp(logits-max(logits)); target=target/sum(target);
adjustedPrevious=baseShare.*x(2:5)./max(previousBaseShare,1e-12);adjustedPrevious=adjustedPrevious/sum(adjustedPrevious);
shares=(1-e.share_partial_adjustment)*adjustedPrevious+e.share_partial_adjustment*target;
if strcmp(rules.consumer_choice,'frozen_initial'), shares=p.initial_shares(:);
elseif strcmp(rules.consumer_choice,'exogenous_trend'), shares=baseShare; end
twhPerMtce=29.3076/3.6;
directRequest=service*shares(1:3)./e.direct_use_efficiency(1:3)'/twhPerMtce;
demand=service*shares(4)/e.direct_use_efficiency(4);
% Physical state ledger: prior projects commission before this year decisions.
queue=reshape(x(21:200),15,12)'; commissioned=queue(:,1);
oldassets=[x(6:14);x(15:17)]; retired=oldassets.*e.retirement_fraction(:);
assets=max(oldassets-retired+commissioned,0); capacities=assets(1:9); productionCapacity=assets(10:12);
q=[queue(:,2:15),zeros(12,1)];
shock=any(year==getv(s,'annual_shock_years',[]));
utilReference=p.utilisation(:);
if p.base_year==2024
    if elapsed==1, utilReference(7)=x(211); else, utilReference(7)=208.3/(45.99*8.76); end
end
potential=capacities(1:7).*utilReference*8.76;
biomassFuelBudget=p.initial_biomass_generation_twh*exp(getv(s,'biomass_feedstock_growth_per_year',.02)*elapsed);
potential(7)=min(potential(7),biomassFuelBudget);
if shock, potential(4:6)=potential(4:6)*(1-getv(s,'renewable_shock_fraction',0)); end
vreLimit=min(.95,e.annual_vre_acceptance_base+e.annual_vre_acceptance_storage_gain*(capacities(8)+capacities(9))/max(demand/5,1));
acceptedVre=min(sum(potential(5:6)),demand*vreLimit);
nf=potential; nf(1:2)=0;
if sum(potential(5:6))>0, nf(5:6)=nf(5:6)*acceptedVre/sum(potential(5:6)); end
if sum(nf)>demand, nf=nf*demand/sum(nf); end
thermalRequest=zeros(2,1); remaining=max(demand-sum(nf),0);
operating=[300;600]+price*[.82;.37]; [~,merit]=sort(operating);
for z=1:2, j=merit(z); thermalRequest(j)=min(remaining,potential(j)); remaining=remaining-thermalRequest(j); end
requested=directRequest+[thermalRequest(1)/(.4*twhPerMtce);0;thermalRequest(2)/(.55*twhPerMtce)];
domesticAvailable=productionCapacity.*p.domestic_retention_fraction(:)*getv(s,'supply_multiplier',1);
importCapacity=x(214:216)*max(gdpRatio,1);
imports=importCapacity*getv(s,'import_multiplier',1);
if shock, imports=imports*(1-getv(s,'import_shock_fraction',.5)); end
ordinary=domesticAvailable+imports;
release=min(x(18:20)*e.max_stock_release_fraction_per_year,max(requested-ordinary,0));
delivered=min(requested,ordinary+release); ratio=min(delivered./max(requested,1e-12),1);
directDelivered=directRequest.*ratio; generation=nf;
generation(1)=thermalRequest(1)*ratio(1); generation(2)=thermalRequest(2)*ratio(3);
powerfuel=[generation(1)/(.4*twhPerMtce);0;generation(2)/(.55*twhPerMtce)];
stockTarget=requested*e.reserve_days*getv(s,'reserve_multiplier',1)/365;
recharge=min(max(stockTarget-(x(18:20)-release),0),requested*e.max_recharge_fraction_of_demand);
recharge=min(recharge,max(ordinary-requested,0)); if shock, recharge=recharge*0; end
used=directDelivered+powerfuel; inflow=used+recharge-release;
actualImports=min(imports,inflow); domestic=inflow-actualImports;
stocks=x(18:20)+domestic+actualImports-used;
stocks(abs(stocks)<1e-10)=0;
unserved=max(demand-sum(generation),0); curtail=sum(potential(3:7)-generation(3:7));
shortage=[1-ratio;unserved/max(demand,1e-12)];
utilisation=generation./max(capacities(1:7)*8.76,1e-12);
gross=used'*p.emission_factors(:); nfPrimary=sum(generation(3:7))*p.nf_primary_factor;
primary=sum(used)+nfPrimary;
industry=p.base_industry*(service/p.initial_service)*exp(-.01*elapsed);
total=gross+industry; net=total-e.sink_mt;
costs=e.capital_cost_million_cny(:);
techlevel=getv(s,'technology_cost_multiplier',1)*exp(-getv(s,'cost_decline_per_year',.015)*elapsed);
costs(5:8)=costs(5:8)*techlevel;
base=p.base_orders(:);delays=ceil(e.construction_years(:)*getv(s,'construction_delay_multiplier',1));
referenceFuture=p.reference_capacity0(:)+p.reference_net_increment(:).*(elapsed+delays(1:7));
referencePrevious=p.reference_capacity0(:)+p.reference_net_increment(:).*(elapsed+delays(1:7)-1);
base(1:7)=max(referenceFuture-(1-e.retirement_fraction(1:7)').*referencePrevious,0);
base(10:12)=max(productionCapacity.*(expm1(p.production_growth(:))+e.retirement_fraction(10:12)'),0);
relative=ones(12,1);
if any(generation(1:2)>1e-6),marginalCost=max(operating(generation(1:2)>1e-6));else,marginalCost=e.no_thermal_market_price_cny_mwh;end
if strcmp(rules.market_price_rule,'step')
    if unserved>1e-6,marketPrice=e.market_scarcity_price_cny_mwh;else,marketPrice=marginalCost;end
else
    fraction=unserved/max(demand,1e-12);
    marketPrice=marginalCost+max(e.market_scarcity_price_cny_mwh-marginalCost,0)*fraction/(fraction+.01);
end
variableCost=[300;600;20;15;5;5;250];
carbonPerTwh=[p.emission_factors(1)/(.4*twhPerMtce);p.emission_factors(3)/(.55*twhPerMtce);0;0;0;0;0];
revenue=generation*marketPrice;
operatingExpense=generation.*(variableCost+price*carbonPerTwh);
annualCapital=costs(1:7).*capacities(1:7)*.08;
profit=revenue-operatingExpense-annualCapital;
referenceCost=e.capital_cost_million_cny(1:7)';referenceCost(5:6)=referenceCost(5:6)*exp(-expectedDecline*elapsed);
referenceProfit=generation.*(marketPrice-variableCost-expectedCarbon*carbonPerTwh)-referenceCost.*capacities(1:7)*.08;
profitRatio=(profit-referenceProfit)./max(referenceCost.*capacities(1:7)*.08,1);
if strcmp(rules.investor_profit_rule,'origin_anchored')
    % Fixed origin information, all monetary terms in million CNY/year.
    anchor=p.investment_reference;initialCost=anchor.capital_cost_million_cny_per_gw(:);
    referencePerGw=8.76*anchor.utilisation(:).*(anchor.market_price_cny_mwh-variableCost-anchor.carbon_price_cny_t*carbonPerTwh)-.08*initialCost;
    referenceProfit=capacities(1:7).*referencePerGw;
    realisedForSignal=generation.*(marketPrice-variableCost-price*carbonPerTwh)-.08*costs(1:7).*capacities(1:7);
    profitRatio=(realisedForSignal-referenceProfit)./max(.08*initialCost.*capacities(1:7),1);
end
if getv(s,'cost_response',true) && rules.investor_profit_response
    relative(1:7)=min(max(1+.25*profitRatio,0),2);
end
feedback=zeros(12,1);
if getv(s,'investment_feedback',true) && rules.investor_utilisation_response
    stress=unserved/max(demand,1)+x(204);
    previousElectricity=x(1)*x(5)/e.direct_use_efficiency(4);
    expectedOneStep=v3_annual_forecast_level(previousElectricity,gdpRatio,1,p.expectations.electricity_fit);
    previousGeneration=x(205:211).*x(6:12)*8.76;
    expectedGeneration=previousGeneration*expectedOneStep/max(previousElectricity,1e-12);
    innovation=(generation-expectedGeneration)./max(capacities(1:7)*8.76,1e-12);
    feedback(1:7)=p.expectations.utilisation_response*capacities(1:7).*innovation+e.investment_feedback*capacities(1:7)*stress;
    feedback(5:6)=feedback(5:6)-e.investment_feedback*capacities(5:6)*curtail/max(sum(potential(5:6)),1);
    feedback(8)=e.investment_feedback*(curtail+unserved)/max(4*.365,1)/20;
    feedback(9)=feedback(8)*.15;
    feedback(10:12)=e.investment_feedback*productionCapacity.*(1-ratio);
end
% Investor agents: paid priorities remain subject to original resource/build caps.
wanted=max(base+feedback,0).*relative;
priority=getv(s,'investment_priority',ones(12,1));wanted=wanted.*priority(:);
% Keep native gross-order expectation anchor; shorten only actual new-project lead.
if strcmp(rules.construction_years,'minimum_one_year'),delays=ones(12,1);end
biomassFutureBudget=p.initial_biomass_generation_twh*exp(getv(s,'biomass_feedstock_growth_per_year',.02)*(elapsed+delays(7)));
biomassUncommitted=max(0,biomassFutureBudget/(utilReference(7)*8.76)-capacities(7)-sum(q(7,:)));
wanted(7)=min(wanted(7),biomassUncommitted);
limit=max(oldassets.*[.15;.20;.15;.08;.40;.50;.20;.50;.15;.15;.10;.20],base*1.5);
wanted=min(wanted,limit);
budget=p.base_budget*(service/p.initial_service)^.7*getv(s,'annual_budget_multiplier',1);
if year<=2030,period=1;elseif year<=2040,period=2;else,period=3;end
periodBudget=getv(s,'budget_multipliers_by_period',[1 1 1]);budget=budget*periodBudget(period);
spend=wanted'*costs; orders=wanted*min(1,budget/max(spend,1e-12)); spend=orders'*costs;
delayfactor=getv(s,'construction_delay_multiplier',1);
delays=ceil(e.construction_years(:)*delayfactor);
if strcmp(rules.construction_years,'minimum_one_year'),delays=ones(12,1);end
assert(all(delays>=1 & delays<=15),'V6:ConstructionDelay','Construction delay outside supported 1-15 years');
for j=1:12, q(j,delays(j))=q(j,delays(j))+orders(j); end
n(1)=service; n(2:5)=shares; n(6:14)=capacities; n(15:17)=productionCapacity; n(18:20)=stocks;
n(21:200)=reshape(q',180,1); n(201:204)=shortage; n(205:211)=utilisation; n(212)=curtail;
carbonFeedback=0;
reference=p.base_national_co2*exp(-getv(s,'carbon_reference_decline_per_year',.02)*max(year-2030,0));
% Government agent: actual emissions error changes next-year carbon state.
if getv(s,'carbon_feedback',true) && rules.carbon_feedback, carbonFeedback=e.carbon_feedback_gain_cny_t*(total/reference-1); end
n(213)=min(e.carbon_price_max_cny_t,max(0,x(213)+getv(s,'carbon_price_drift_cny_t',3)+carbonFeedback));
n(214:216)=importCapacity; n(217)=marketPrice;
metrics=[demand;sum(generation);unserved;primary;gross;net;max(net,0);curtail;spend;budget;nfPrimary;total;price;service;generation;directRequest;directDelivered;powerfuel;domestic;actualImports;stocks;stocks-x(18:20);orders;commissioned;retired;feedback;costs;profit;revenue;operatingExpense;annualCapital;marketPrice;reference];
out=[n;metrics];
v3_live_monitor('annual',year,n,metrics);
end

function value=getv(s,key,default)
if isfield(s,key), value=s.(key); else, value=default; end
end
