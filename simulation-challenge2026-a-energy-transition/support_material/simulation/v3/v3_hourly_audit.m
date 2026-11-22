function checks = v3_hourly_audit(c,T)
% Independent reconstruction from raw inputs, actions and realized end states.
r=struct; r.all_finite=all(isfinite(T{:,:}),'all');
generation=T{:,cellstr(c.generators+"_gw")};
r.generation_bound_gw=max([0;-generation(:);reshape(generation-c.available_gw,[],1)]);
r.biomass_feedstock_budget_gwh=max(0,sum(T.biomass_gw)-c.biomass_generation_budget_twh_per_year*1000*c.hours/8760);
balance=sum(generation,2)+T.unserved_gw-T.spill_gw-c.demand_gw+T.shift_out_gw-T.shift_in_gw;
for j=1:3
    f=c.fuels(j); states=[c.initial_fuel_stocks_mtce(j);T.(f+"_stock_mtce")];
    short=T.(f+"_direct_unserved_mtce"); delivered=c.direct_fuel_demand_mtce(:,j)-short;
    burn=T.(f+"_power_burn_mtce");
    if f=="coal"
        expected=T.coal_gw/(8141*c.fuel_efficiency.coal);
    elseif f=="gas"
        expected=T.gas_gw/(8141*c.fuel_efficiency.gas);
    else
        expected=zeros(c.hours,1);
    end
    r.(f+"_balance_mtce")=max(abs(diff(states)-c.fuel_arrivals_mtce(:,j)+delivered+expected));
    r.(f+"_bounds_mtce")=max([0;-states;-short;short-c.direct_fuel_demand_mtce(:,j)]);
    r.(f+"_export_error_mtce")=max([abs(burn-expected);abs(T.(f+"_direct_served_mtce")-delivered); ...
        abs(T.(f+"_arrival_mtce")-c.fuel_arrivals_mtce(:,j)); ...
        abs(T.(f+"_direct_demand_mtce")-c.direct_fuel_demand_mtce(:,j))]);
end
for j=1:2
    name=c.stores(j); st=c.storage.(name);
    ch=T.(name+"_charge_gw"); dis=T.(name+"_discharge_gw");
    soc=[st.initial_soc_gwh;T.(name+"_soc_gwh")];
    r.(name+"_state_gwh")=max(abs(diff(soc)-st.eta_charge*ch+dis/st.eta_discharge));
    r.(name+"_bounds")=max([0;-soc;soc-st.energy_gwh;-ch;-dis;ch+dis-st.power_gw]);
    balance=balance+dis-ch;
end
r.power_balance_gw=max(abs(balance));
debt=[0;T.shift_debt_gwh];
r.debt_state_gwh=max(abs(diff(debt)-T.shift_out_gw+T.shift_in_gw));
deadline=mod((1:c.hours)',c.config.deadline_hours)==0; deadline(end)=true;
r.debt_deadline_gwh=max(abs(T.shift_debt_gwh(deadline)));
r.flex_bounds=max([0;-debt;-T.shift_out_gw;-T.shift_in_gw; ...
    T.shift_out_gw-c.config.flex_fraction*c.demand_gw; ...
    T.shift_in_gw-c.config.repay_fraction*c.demand_gw]);
r.slack_bounds=max([0;-T.unserved_gw;-T.spill_gw;T.unserved_gw+T.shift_out_gw-c.demand_gw]);
for j=1:2
    f=c.generators(j); previous=[c.initial_thermal_gw(j);T.(f+"_gw")];
    available=[c.available_gw(1,j);c.available_gw(:,j)];
    ramp=c.capacity_gw.(f)*c.config.ramp_fraction(j);
    r.(f+"_ramp_gw")=max([0;diff(previous)-ramp;-diff(previous)-max(ramp,max(0,-diff(available)))]);
    r.(f+"_ramp_memory_gw")=max(abs(T.(f+"_ramp_state_gw")-T.(f+"_gw")));
end
r.served_power_export_gw=max(abs(T.served_power_gw-(c.demand_gw-T.shift_out_gw+T.shift_in_gw-T.unserved_gw)));
values=struct2array(r); checks=r; checks.tolerance=1e-5;
checks.passed=r.all_finite && all(values(2:end)<=checks.tolerance);
checks.interpretation="Independent state/action reconstruction, not an empirical hourly validation claim.";
end
