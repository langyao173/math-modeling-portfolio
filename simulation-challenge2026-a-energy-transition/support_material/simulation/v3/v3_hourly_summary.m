function s = v3_hourly_summary(c,T,strategy,runtime)
% Resource accounting charges consumed initial assets and credits final SOC.
cfg=c.config; ef=[94.6,73.3,56.1]*29.3076/1000;
electricCo2=T.coal_power_burn_mtce*ef(1)+T.gas_power_burn_mtce*ef(3);
directCo2=0; arrivalValue=0; stockDrawValue=0; directShort=0;
s=struct('scenario',c.scenario,'year',c.year,'shock',c.shock, ...
    'strategy',"native_"+strategy+"_mpc",'hours',c.hours,'forecast_hours',24, ...
    'unserved_gwh',sum(T.unserved_gw),'peak_unserved_gw',max(T.unserved_gw), ...
    'loss_of_load_hours',sum(T.unserved_gw>1e-5), ...
    'shift_out_gwh',sum(T.shift_out_gw),'shift_in_gwh',sum(T.shift_in_gw), ...
    'terminal_shift_debt_gwh',T.shift_debt_gwh(end),'runtime_seconds',runtime);
for j=1:3
    f=c.fuels(j); price=cfg.fuel_value_million_cny_mtce(j);
    directCo2=directCo2+sum(T.(f+"_direct_served_mtce"))*ef(j);
    arrivalValue=arrivalValue+sum(c.fuel_arrivals_mtce(:,j))*price;
    stockDrawValue=stockDrawValue+(c.initial_fuel_stocks_mtce(j)-T.(f+"_stock_mtce")(end))*price;
    short=sum(T.(f+"_direct_unserved_mtce")); directShort=directShort+short;
    s.(f+"_direct_unserved_mtce")=short;
    s.(f+"_initial_stock_mtce")=c.initial_fuel_stocks_mtce(j);
    s.(f+"_terminal_stock_mtce")=T.(f+"_stock_mtce")(end);
    empty=find(T.(f+"_stock_mtce")<1e-7,1);
    if isempty(empty), s.(f+"_first_empty_hour")=NaN;
    else, s.(f+"_first_empty_hour")=empty-1; end
end
socDrawValue=0; storageCost=0;
for j=1:2
    name=c.stores(j); st=c.storage.(name);
    socDrawValue=socDrawValue+(st.initial_soc_gwh-T.(name+"_soc_gwh")(end))*cfg.stored_electricity_value_million_cny_gwh;
    storageCost=storageCost+sum(T.(name+"_charge_gw")+T.(name+"_discharge_gw"))*cfg.storage_cycle_cost_cny_kwh;
    s.(name+"_initial_soc_gwh")=st.initial_soc_gwh;
    s.(name+"_terminal_soc_gwh")=T.(name+"_soc_gwh")(end);
end
nonfuel=0;
for j=3:7, nonfuel=nonfuel+sum(T.(c.generators(j)+"_gw"))*cfg.generation_cost_cny_kwh(j); end
fuelCost=arrivalValue+stockDrawValue+nonfuel;
shiftCost=sum(T.shift_out_gw+T.shift_in_gw)*cfg.shift_cost_cny_kwh;
shortageCost=sum(T.unserved_gw)*cfg.unserved_cost_cny_kwh+directShort*8141*cfg.direct_shortage_cost_cny_kwh_th;
carbonCost=(sum(electricCo2)+directCo2)*cfg.carbon_price_cny_t;
s.direct_fuel_unserved_mtce=directShort;
s.electricity_co2_mt=sum(electricCo2); s.direct_fuel_co2_mt=directCo2;
s.total_co2_mt=sum(electricCo2)+directCo2;
s.fuel_consumption_and_nonfossil_cost_million_cny=fuelCost;
s.storage_cost_million_cny=storageCost; s.shift_cost_million_cny=shiftCost;
s.shortage_penalty_million_cny=shortageCost; s.carbon_cost_million_cny=carbonCost;
s.total_objective_components_million_cny=fuelCost+storageCost+shiftCost+shortageCost+carbonCost;
s.fuel_arrival_purchase_value_million_cny=arrivalValue;
s.net_fuel_inventory_draw_value_million_cny=stockDrawValue;
s.net_soc_draw_value_million_cny=socDrawValue;
s.resource_accounted_cost_million_cny=s.total_objective_components_million_cny+socDrawValue;
s.initial_fuel_inventory_value_million_cny=sum(c.initial_fuel_stocks_mtce.*cfg.fuel_value_million_cny_mtce);
s.terminal_fuel_inventory_value_million_cny=s.initial_fuel_inventory_value_million_cny-stockDrawValue;
s.stock_holding_cost_million_cny=s.initial_fuel_inventory_value_million_cny*cfg.stock_holding_rate_per_year*c.hours/8760;
s.resource_plus_holding_cost_million_cny=s.resource_accounted_cost_million_cny+s.stock_holding_cost_million_cny;
s.stock_holding_assumption="5 percent/year on committed initial fuel inventory value, prorated; declared capital charge, not observed tariff or optimized hourly cost.";
s.valuation_scope="Declared constant engineering values, not market price observations. Arrival purchases plus initial-minus-terminal fuel assets equal actual consumption; gas/coal values reproduce their per-GWh costs. SOC draw charged at common 0.32 million CNY/GWh.";
s.information_contract=c.information_contract; s.evidence_class=c.evidence_class; s.boundary=c.boundary;
end
