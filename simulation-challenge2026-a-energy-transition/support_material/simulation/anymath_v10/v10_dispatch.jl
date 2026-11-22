# AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
# Reoptimized actions, not a playback of MATLAB controls. Physical state may be
# advanced by an AnyMath plant callback, whose returned state feeds the next LP.
using JuMP, HiGHS, CSV, DataFrames
include(joinpath(@__DIR__, "v10_config.jl"))

function v10_case(config=v10_config(); input_path=joinpath(@__DIR__, "v10_base_inputs.csv"))
    @assert 2 <= config.hours <= 2160
    @assert length(config.event_starts)==2 && 0 <= config.shock_severity <= 1
    @assert config.storage_power_multiplier > 0 && config.contract_multiplier >= 0
    @assert config.supply_policy in ["inherited", "contract"]
    @assert config.strategy in ["coordinated", "decentralized"]
    s=config.event_starts; d=config.event_duration_hours; H=config.hours
    @assert 0 <= s[1] < s[1]+d < s[2] < s[2]+d < H
    @assert config.contract_lead_hours >= 0
    raw=CSV.read(input_path, DataFrame)[1:H,:]
    @assert raw.hour == collect(0:H-1)
    demand=Float64.(raw.demand_gw)
    available=Matrix{Float64}(raw[:,[g*"_available_gw" for g in V10_GENERATORS]])
    domestic=Matrix{Float64}(raw[:,[f*"_domestic_mtce" for f in V10_FUELS]])
    imports=Matrix{Float64}(raw[:,[f*"_import_mtce" for f in V10_FUELS]])
    direct=Matrix{Float64}(raw[:,[f*"_direct_demand_mtce" for f in V10_FUELS]])
    @assert all(isfinite, [demand; vec(available); vec(domestic); vec(imports); vec(direct)])
    @assert minimum([demand; vec(available); vec(domestic); vec(imports); vec(direct)]) >= 0
    stress=falses(H); emergency=zeros(H,3)
    for (event,start) in enumerate(s)
        stop=start+d
        rows=(start+1):stop
        stress[rows].=true
        imports[rows,:].*=reshape(1 .- config.shock_severity .* (1 .- [.50,.05,.05]),1,3)
        for (column,retention) in [(5,.25),(6,.55),(4,.70)]
            available[rows,column].*=1-config.shock_severity*(1-retention)
        end
        demand[rows].*=1+.08*config.shock_severity
        if config.supply_policy=="contract"
            first=stop+config.contract_lead_hours
            last=event<2 ? s[event+1] : H
            @assert first < last "Contract requires a positive recovery delivery window"
            rate=min.(V10_IMPORT_CAPACITY_PER_H .* .5 .* config.contract_multiplier,
                V10_INITIAL_FUEL .* .5 .* config.contract_multiplier ./ (last-first))
            emergency[first+1:last,:].+=reshape(rate,1,3)
        end
    end
    return (;config, demand, available, arrivals=domestic+imports+emergency, direct,
        domestic, imports, emergency, stress,
        storage_power=V10_STORAGE_POWER .* config.storage_power_multiplier)
end

function v10_local_storage(c,first,n,state)
    net=c.demand[first:first+n-1]-vec(sum(c.available[first:first+n-1,3:7],dims=2))
    ch=zeros(n,2); dis=zeros(n,2)
    for j in 1:2
        soc=state.soc[j]
        for k in 1:n
            if net[k]<0
                ch[k,j]=min(-net[k],c.storage_power[j],max(0,V10_STORAGE_ENERGY[j]-soc)/V10_ETA[j])
                soc+=ch[k,j]*V10_ETA[j];net[k]+=ch[k,j]
            else
                dis[k,j]=min(net[k],c.storage_power[j],max(0,soc)*V10_ETA[j])
                soc-=dis[k,j]/V10_ETA[j];net[k]-=dis[k,j]
            end
        end
    end
    return (;ch,dis)
end

"""One common 24h perfect-forecast LP. `first` is one-based; states are pre-hour.
`action` is gen(7), ch(2), dis(2), direct_short(3), out, repay, unserved, spill.
Fuel Mtce, power GW, energy/debt GWh, cost million CNY. Only hour one executes.
"""
function v10_plan(c,first,state)
    n=min(24,c.config.hours-first+1)
    @assert n>0
    model=Model(HiGHS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model,"threads",1)
    @variable(model,gen[1:n,1:7]>=0)
    @variable(model,ch[1:n,1:2]>=0)
    @variable(model,dis[1:n,1:2]>=0)
    @variable(model,soc[1:n,1:2]>=0)
    @variable(model,fuel[1:n,1:3]>=0)
    @variable(model,direct_short[1:n,1:3]>=0)
    @variable(model,out[1:n]>=0)
    @variable(model,repay[1:n]>=0)
    @variable(model,debt[1:n]>=0)
    @variable(model,unserved[1:n]>=0)
    @variable(model,spill[1:n]>=0)
    local_plan=c.config.strategy=="decentralized" ? v10_local_storage(c,first,n,state) : nothing
    objective=AffExpr(0.0)
    for k in 1:n
        t=first+k-1; urgency=1+1e-5*(n-k+1)/n
        for j in 1:7
            set_upper_bound(gen[k,j],c.available[t,j])
            carbon=j<=2 ? V10_FUEL_EMISSION[j==1 ? 1 : 3]/(8141*V10_EFFICIENCY[j])*V10_CARBON_PRICE : 0.0
            add_to_expression!(objective,V10_GENERATION_COST[j]+carbon,gen[k,j])
        end
        for j in 1:3
            set_upper_bound(direct_short[k,j],c.direct[t,j])
            add_to_expression!(objective,30*8141*urgency-V10_FUEL_VALUE[j]-V10_FUEL_EMISSION[j]*V10_CARBON_PRICE,direct_short[k,j])
        end
        for j in 1:2
            set_upper_bound(soc[k,j],V10_STORAGE_ENERGY[j])
            set_upper_bound(ch[k,j],c.storage_power[j]);set_upper_bound(dis[k,j],c.storage_power[j])
            add_to_expression!(objective,.005,ch[k,j]);add_to_expression!(objective,.005,dis[k,j])
            if k==n;add_to_expression!(objective,-.32,soc[k,j]);end
            if local_plan!==nothing
                fix(ch[k,j],local_plan.ch[k,j];force=true)
                fix(dis[k,j],local_plan.dis[k,j];force=true)
            end
        end
        set_upper_bound(out[k],.08*c.demand[t]);set_upper_bound(repay[k],.20*c.demand[t])
        add_to_expression!(objective,.003,out[k]);add_to_expression!(objective,.003,repay[k])
        add_to_expression!(objective,30*urgency,unserved[k])
        if t%24==0 || t==c.config.hours;set_upper_bound(debt[k],0.0);end
        @constraint(model,sum(gen[k,:])+sum(dis[k,:])-sum(ch[k,:])+out[k]-repay[k]+unserved[k]-spill[k]==c.demand[t])
        @constraint(model,unserved[k]+out[k]<=c.demand[t])
        for j in 1:3
            previous=k==1 ? state.fuel[j] : fuel[k-1,j]
            burn=j==1 ? gen[k,1]/(8141*V10_EFFICIENCY[1]) : (j==3 ? gen[k,2]/(8141*V10_EFFICIENCY[2]) : 0.0)
            @constraint(model,fuel[k,j]==previous+c.arrivals[t,j]-c.direct[t,j]+direct_short[k,j]-burn)
        end
        for j in 1:2
            previous=k==1 ? state.soc[j] : soc[k-1,j]
            @constraint(model,soc[k,j]==previous+V10_ETA[j]*ch[k,j]-dis[k,j]/V10_ETA[j])
            @constraint(model,ch[k,j]+dis[k,j]<=c.storage_power[j])
        end
        previous_debt=k==1 ? state.debt : debt[k-1]
        @constraint(model,debt[k]==previous_debt+out[k]-repay[k])
        for j in 1:2
            previous=k==1 ? state.thermal[j] : gen[k-1,j]
            previous_available=c.available[max(1,t-1),j]
            drop=max(0,previous_available-c.available[t,j]);ramp=V10_THERMAL_CAPACITY[j]*[.15,.70][j]
            @constraint(model,gen[k,j]-previous<=ramp)
            @constraint(model,previous-gen[k,j]<=max(ramp,drop))
        end
    end
    @objective(model,Min,objective)
    optimize!(model)
    status=termination_status(model)
    @assert status==MOI.OPTIMAL "LP hour $(first-1) failed: $status"
    action=(gen=value.(gen[1,:]),ch=value.(ch[1,:]),dis=value.(dis[1,:]),
        direct_short=value.(direct_short[1,:]),out=value(out[1]),repay=value(repay[1]),
        unserved=value(unserved[1]),spill=value(spill[1]))
    expected=(fuel=value.(fuel[1,:]),soc=value.(soc[1,:]),thermal=action.gen[1:2],debt=value(debt[1]))
    return (;action,expected,objective=objective_value(model),status=string(status),forecast_hours=n)
end

v10_initial_state()=(fuel=copy(V10_INITIAL_FUEL),soc=copy(V10_INITIAL_SOC),thermal=zeros(2),debt=0.0)

"""Independent Julia conservation, used only when no native plant is attached.
AnyMath integration replaces this function with the actual native execution hook.
"""
function v10_accounting_plant(state,a,e)
    burn=[a.gen[1]/(8141*V10_EFFICIENCY[1]),0.0,a.gen[2]/(8141*V10_EFFICIENCY[2])]
    return (fuel=state.fuel+e.arrivals-e.direct+a.direct_short-burn,
        soc=state.soc+V10_ETA.*a.ch-a.dis./V10_ETA,
        thermal=copy(a.gen[1:2]),debt=state.debt+a.out-a.repay)
end

mutable struct V10Session
    case::NamedTuple
    state::NamedTuple
    next_hour::Int
    rows::Vector{NamedTuple}
    audit_rows::Vector{NamedTuple}
    plant_backend::String
    elapsed_seconds::Float64
end

function v10_session(config=v10_config(); input_path=joinpath(@__DIR__,"v10_base_inputs.csv"),
    plant_backend="JULIA_ACCOUNTING_NOT_NATIVE")
    return V10Session(v10_case(config;input_path),v10_initial_state(),0,NamedTuple[],NamedTuple[],plant_backend,0.0)
end

function v10_step_audit(c,t,state,a,new,expected)
    e=(arrivals=c.arrivals[t,:],direct=c.direct[t,:])
    reconstructed=v10_accounting_plant(state,a,e)
    accounting=max(maximum(abs.(new.fuel-reconstructed.fuel)),maximum(abs.(new.soc-reconstructed.soc)),
        maximum(abs.(new.thermal-reconstructed.thermal)),abs(new.debt-reconstructed.debt))
    predicted=max(maximum(abs.(new.fuel-expected.fuel)),maximum(abs.(new.soc-expected.soc)),abs(new.debt-expected.debt))
    power=abs(sum(a.gen)+sum(a.dis)-sum(a.ch)+a.out-a.repay+a.unserved-a.spill-c.demand[t])
    bounds=max(0.0,-minimum([a.gen;a.ch;a.dis;a.direct_short;a.out;a.repay;a.unserved;a.spill;new.fuel;new.soc;new.debt]),
        maximum(a.gen-c.available[t,:]),maximum(a.ch+a.dis-c.storage_power),
        maximum(new.soc-V10_STORAGE_ENERGY),maximum(a.direct_short-c.direct[t,:]),
        a.out-.08*c.demand[t],a.repay-.20*c.demand[t],a.unserved+a.out-c.demand[t])
    ramp=V10_THERMAL_CAPACITY.*[.15,.70]
    drop=max.(0,c.available[max(1,t-1),1:2]-c.available[t,1:2])
    ramp_violation=max(0.0,maximum(a.gen[1:2]-state.thermal-ramp),maximum(state.thermal-a.gen[1:2]-max.(ramp,drop)))
    deadline=(t%24==0 || t==c.config.hours) ? abs(new.debt) : 0.0
    @assert max(accounting,predicted,power,bounds,ramp_violation,deadline)<1e-6 "Physical check failed at hour $(t-1)"
    return (;hour=t-1,accounting_residual=accounting,lp_state_residual=predicted,
        power_residual_gw=power,bound_violation=bounds,ramp_violation_gw=ramp_violation,debt_deadline_gwh=deadline)
end

"""Execute n actual hourly solves. Hook: plant_step(prior,action,external) ->
(fuel[3],soc[2],thermal[2],debt). The next LP reads exactly that returned state.
Changing parameters requires a fresh session; existing observations never change.
"""
function step_v10!(session,n=24;plant_step=v10_accounting_plant)
    @assert n>=1
    started=time();c=session.case
    for hour in session.next_hour:min(session.next_hour+n-1,c.config.hours-1)
        t=hour+1;prior=session.state
        solved=v10_plan(c,t,prior);a=solved.action
        external=(;hour,arrivals=c.arrivals[t,:],direct=c.direct[t,:],available=c.available[t,:],demand=c.demand[t])
        new=plant_step(prior,a,external)
        checks=v10_step_audit(c,t,prior,a,new,solved.expected)
        row=(;hour,demand_gw=c.demand[t],stress_active=Int(c.stress[t]),
            shift_out_gw=a.out,shift_in_gw=a.repay,unserved_gw=a.unserved,spill_gw=a.spill,
            (Symbol(g*"_gw")=>a.gen[j] for (j,g) in enumerate(V10_GENERATORS))...,
            (Symbol(g*"_available_gw")=>c.available[t,j] for (j,g) in enumerate(V10_GENERATORS))...,
            (Symbol(f*"_stock_mtce")=>new.fuel[j] for (j,f) in enumerate(V10_FUELS))...,
            (Symbol(f*"_arrival_mtce")=>c.arrivals[t,j] for (j,f) in enumerate(V10_FUELS))...,
            (Symbol(f*"_emergency_mtce")=>c.emergency[t,j] for (j,f) in enumerate(V10_FUELS))...,
            (Symbol(f*"_direct_demand_mtce")=>c.direct[t,j] for (j,f) in enumerate(V10_FUELS))...,
            (Symbol(f*"_direct_unserved_mtce")=>a.direct_short[j] for (j,f) in enumerate(V10_FUELS))...,
            battery_charge_gw=a.ch[1],battery_discharge_gw=a.dis[1],battery_soc_gwh=new.soc[1],
            pumped_charge_gw=a.ch[2],pumped_discharge_gw=a.dis[2],pumped_soc_gwh=new.soc[2],
            shift_debt_gwh=new.debt,served_power_gw=c.demand[t]-a.out+a.repay-a.unserved,
            lp_objective_million_cny=solved.objective,lp_status=solved.status,forecast_hours=solved.forecast_hours,
            plant_backend=session.plant_backend)
        push!(session.rows,row);push!(session.audit_rows,checks)
        session.state=new;session.next_hour=hour+1
    end
    session.elapsed_seconds+=time()-started
    return DataFrame(session.rows)
end

function v10_recovery(session)
    c=session.case;table=DataFrame(session.rows);completed=nrow(table);result=NamedTuple[]
    for (event,start) in enumerate(c.config.event_starts)
        stop=start+c.config.event_duration_hours
        last=event<2 ? c.config.event_starts[event+1] : c.config.hours
        observed_last=min(last,completed)
        if completed<stop;continue;end
        pre=start==0 ? V10_INITIAL_FUEL : [table[start,Symbol(f*"_stock_mtce")] for f in V10_FUELS]
        target=0.95 .* pre
        service_start=missing;joint_start=missing
        # Same definition as V6: 24h energy-weighted service, stocks at every
        # end-of-hour in that window. A window is assessed only after it exists.
        for first in stop+1:observed_last-23
            window=first:first+23
            service_ok=sum(table.unserved_gw[window])<=.01*sum(table.demand_gw[window])+1e-8 && all(
                sum(table[window,Symbol(f*"_direct_unserved_mtce")])<=.01*sum(c.direct[window,j])+1e-8
                for (j,f) in enumerate(V10_FUELS))
            stock_ok=all(all(table[window,Symbol(f*"_stock_mtce")].>=target[j]-1e-8) for (j,f) in enumerate(V10_FUELS))
            if service_ok && ismissing(service_start);service_start=first-1;end
            if service_ok && stock_ok && ismissing(joint_start);joint_start=first-1;end
        end
        boundary_stocks=[table[observed_last,Symbol(f*"_stock_mtce")] for f in V10_FUELS]
        boundary_ratios=[pre[j]>1e-12 ? boundary_stocks[j]/pre[j] : missing for j in 1:3]
        final_window_ready=false
        if observed_last>=stop+24
            window=observed_last-23:observed_last
            final_window_ready=sum(table.unserved_gw[window])<=.01*sum(table.demand_gw[window])+1e-8 && all(
                sum(table[window,Symbol(f*"_direct_unserved_mtce")])<=.01*sum(c.direct[window,j])+1e-8 &&
                all(table[window,Symbol(f*"_stock_mtce")].>=target[j]-1e-8) for (j,f) in enumerate(V10_FUELS))
        end
        stock_rebreach=missing
        if !ismissing(joint_start)
            for i in joint_start+25:observed_last
                if any(table[i,Symbol(f*"_stock_mtce")]<target[j]-1e-8 for (j,f) in enumerate(V10_FUELS))
                    stock_rebreach=i-1;break
                end
            end
        end
        push!(result,(;event,start_hour=start,shock_end_hour=stop,observation_end_hour=observed_last,
            target_coal_mtce=target[1],target_oil_mtce=target[2],target_gas_mtce=target[3],
            first_service_24h_start=service_start,first_joint_24h_start=joint_start,
            service_confirmation_hour=ismissing(service_start) ? missing : service_start+23,
            joint_confirmation_hour=ismissing(joint_start) ? missing : joint_start+23,
            service_wait_hours=ismissing(service_start) ? missing : service_start-stop,
            joint_wait_hours=ismissing(joint_start) ? missing : joint_start-stop,
            service_right_censored=ismissing(service_start),joint_right_censored=ismissing(joint_start),
            observation_complete=observed_last==last,
            coal_boundary_ratio=boundary_ratios[1],oil_boundary_ratio=boundary_ratios[2],gas_boundary_ratio=boundary_ratios[3],
            final_24h_joint_ready=final_window_ready,stock_rebreach_hour_after_joint_confirmation=stock_rebreach,
            recovery_definition="24h aggregate electricity and each direct-fuel shortage <=1%; all three stocks >=95% preevent at every end-hour"))
    end
    return DataFrame(result)
end

function v10_summary(session)
    c=session.case;table=DataFrame(session.rows);N=nrow(table)
    @assert N>0
    gen=Matrix(table[:,[Symbol(g*"_gw") for g in V10_GENERATORS]])
    direct_short=Matrix(table[:,[Symbol(f*"_direct_unserved_mtce") for f in V10_FUELS]])
    electric_co2=sum(gen[:,1]./(8141*V10_EFFICIENCY[1]))*V10_FUEL_EMISSION[1]+sum(gen[:,2]./(8141*V10_EFFICIENCY[2]))*V10_FUEL_EMISSION[3]
    direct_co2=sum(vec(sum(c.direct[1:N,:]-direct_short,dims=1)).*V10_FUEL_EMISSION)
    final_fuel=[table[N,Symbol(f*"_stock_mtce")] for f in V10_FUELS]
    arrival_value=sum(vec(sum(c.arrivals[1:N,:],dims=1)).*V10_FUEL_VALUE)
    inventory_draw=sum((V10_INITIAL_FUEL-final_fuel).*V10_FUEL_VALUE)
    nonfossil_cost=sum(sum(gen[:,j])*V10_GENERATION_COST[j] for j in 3:7)
    storage_cost=.005*sum(table.battery_charge_gw+table.battery_discharge_gw+table.pumped_charge_gw+table.pumped_discharge_gw)
    shift_cost=.003*sum(table.shift_out_gw+table.shift_in_gw)
    shortage_cost=30*sum(table.unserved_gw)+30*8141*sum(direct_short)
    carbon_cost=(electric_co2+direct_co2)*V10_CARBON_PRICE
    soc_draw=.32*(sum(V10_INITIAL_SOC)-table.battery_soc_gwh[N]-table.pumped_soc_gwh[N])
    holding=sum(V10_INITIAL_FUEL.*V10_FUEL_VALUE)*.05*N/8760
    emergency_value=sum(vec(sum(c.emergency[1:N,:],dims=1)).*V10_FUEL_VALUE)
    resource_cost=arrival_value+inventory_draw+nonfossil_cost+storage_cost+shift_cost+soc_draw
    audit=DataFrame(session.audit_rows)
    return DataFrame([(;design=c.config.design,executed_hours=N,complete=N==c.config.hours,
        plant_backend=session.plant_backend,strategy=c.config.strategy,supply_policy=c.config.supply_policy,
        shock_severity=c.config.shock_severity,event_duration_hours=c.config.event_duration_hours,
        storage_power_multiplier=c.config.storage_power_multiplier,contract_multiplier=c.config.contract_multiplier,
        unserved_gwh=sum(table.unserved_gw),warmup_unserved_gwh=sum(table.unserved_gw[table.hour.<c.config.event_starts[1]]),
        post_warmup_unserved_gwh=sum(table.unserved_gw[table.hour.>=c.config.event_starts[1]]),
        peak_unserved_gw=maximum(table.unserved_gw),loss_of_load_hours=count(table.unserved_gw.>1e-5),
        direct_fuel_unserved_mtce=sum(direct_short),electricity_co2_mt=electric_co2,direct_fuel_co2_mt=direct_co2,
        total_co2_mt=electric_co2+direct_co2,resource_accounted_cost_million_cny=resource_cost,
        shortage_penalty_million_cny=shortage_cost,carbon_cost_million_cny=carbon_cost,
        stock_holding_cost_million_cny=holding,contract_base_value_million_cny=emergency_value,
        contract_premium_million_cny=.5*emergency_value,
        physical_operating_cost_with_contract_premium_million_cny=resource_cost+holding+.5*emergency_value,
        penalty_inclusive_cost_million_cny=resource_cost+holding+.5*emergency_value+shortage_cost+carbon_cost,
        fuel_arrival_purchase_value_million_cny=arrival_value,
        coal_terminal_stock_mtce=final_fuel[1],oil_terminal_stock_mtce=final_fuel[2],gas_terminal_stock_mtce=final_fuel[3],
        terminal_shift_debt_gwh=table.shift_debt_gwh[N],
        maximum_physical_residual=maximum(Matrix(audit[:,Not(:hour)])),
        optimal_lp_hours=count(table.lp_status.=="OPTIMAL"),elapsed_seconds=session.elapsed_seconds,
        information_contract="COMMON_24H_PERFECT_FORECAST_ROLLING_FIRST_ACTION")])
end

function v10_export(session,output_dir;name="run")
    mkpath(output_dir)
    CSV.write(joinpath(output_dir,name*"_actions_states.csv"),DataFrame(session.rows))
    CSV.write(joinpath(output_dir,name*"_physical_audit.csv"),DataFrame(session.audit_rows))
    CSV.write(joinpath(output_dir,name*"_recovery.csv"),v10_recovery(session))
    CSV.write(joinpath(output_dir,name*"_summary.csv"),v10_summary(session))
    cfg=session.case.config
    CSV.write(joinpath(output_dir,name*"_config.csv"),DataFrame([(;design=cfg.design,hours=cfg.hours,
        event1_start_hour=cfg.event_starts[1],event2_start_hour=cfg.event_starts[2],
        event_duration_hours=cfg.event_duration_hours,shock_severity=cfg.shock_severity,
        supply_policy=cfg.supply_policy,contract_multiplier=cfg.contract_multiplier,
        contract_lead_hours=cfg.contract_lead_hours,storage_power_multiplier=cfg.storage_power_multiplier,
        strategy=cfg.strategy)]))
    open(joinpath(output_dir,name*"_config.txt"),"w") do io
        println(io,repr(session.case.config));println(io,"plant_backend=",session.plant_backend)
        println(io,"executed_hours=",session.next_hour);println(io,"elapsed_seconds=",session.elapsed_seconds)
        println(io,"information_contract=COMMON_24H_PERFECT_FORECAST_ROLLING_FIRST_ACTION")
    end
    return DataFrame(session.rows)
end

function run_v10(config=v10_config();output_dir=joinpath(@__DIR__,"v10_results"),name="run",
    plant_step=v10_accounting_plant,plant_backend="JULIA_ACCOUNTING_NOT_NATIVE")
    session=v10_session(config;plant_backend)
    step_v10!(session,config.hours;plant_step)
    v10_export(session,output_dir;name)
    return session
end
