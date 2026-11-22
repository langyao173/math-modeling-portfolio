function T=v6_session_trace(c,evidence)
% Export actual native actions and end states with the frozen exogenous inputs.
n=numel(evidence.hours);assert(isequal(evidence.hours,(0:n-1)'),'V6:Clock','Non-contiguous native prefix.');
columns={'coal_gw','gas_gw','nuclear_gw','hydro_gw','wind_gw','solar_gw','biomass_gw', ...
    'battery_charge_gw','battery_discharge_gw','pumped_charge_gw','pumped_discharge_gw', ...
    'coal_direct_unserved_mtce','oil_direct_unserved_mtce','gas_direct_unserved_mtce', ...
    'shift_out_gw','shift_in_gw','unserved_gw','spill_gw', ...
    'coal_stock_mtce','oil_stock_mtce','gas_stock_mtce','coal_ramp_state_gw','gas_ramp_state_gw', ...
    'battery_soc_gwh','pumped_soc_gwh','shift_debt_gwh'};
T=[table(evidence.hours,c.demand_gw(1:n),double(c.stress_mask(1:n)), ...
    'VariableNames',{'hour','demand_gw','stress_active'}),array2table(evidence.values,'VariableNames',columns)];
for j=1:7,T.(c.generators(j)+"_available_gw")=c.available_gw(1:n,j);end
for j=1:3
    f=c.fuels(j);T.(f+"_arrival_mtce")=c.fuel_arrivals_mtce(1:n,j);
    T.(f+"_direct_demand_mtce")=c.direct_fuel_demand_mtce(1:n,j);
    T.(f+"_direct_served_mtce")=c.direct_fuel_demand_mtce(1:n,j)-T.(f+"_direct_unserved_mtce");
    T.(f+"_power_burn_mtce")=zeros(n,1);
end
T.coal_power_burn_mtce=T.coal_gw/(8141*c.fuel_efficiency.coal);
T.gas_power_burn_mtce=T.gas_gw/(8141*c.fuel_efficiency.gas);
T.served_power_gw=T.demand_gw-T.shift_out_gw+T.shift_in_gw-T.unserved_gw;
end
