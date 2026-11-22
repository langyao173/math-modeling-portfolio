function y = v3_hourly_mpc_step(u,caseInput,strategyInput)
% Each call solves a fresh MATLAB LP from physical Simulink state feedback.
% Persistent data are read-only experiment forecasts, never physical states.
persistent currentCase currentStrategy
if nargin==3
    currentCase=caseInput; currentStrategy=string(strategyInput); y=[]; return;
end
assert(~isempty(currentCase),'Configure native hourly controller before sim.');
c=currentCase; cfg=c.config; first=round(u(1))+1;
% Forecasts are actual visible Simulink input signals (14 channels x 24h).
% Only engineering parameters and experiment identity remain persistent.
raw=reshape(u(2:337),14,24)';
valid=min(cfg.forecast_hours,c.hours-first+1); rr=first:first+valid-1;
c.demand_gw(rr)=raw(1:valid,1);
c.available_gw(rr,:)=raw(1:valid,2:8);
c.fuel_arrivals_mtce(rr,:)=raw(1:valid,9:11);
c.direct_fuel_demand_mtce(rr,:)=raw(1:valid,12:14);
state.fuel=u(338:340)'; state.thermal=u(341:342)';
state.soc=u(343:344)'; state.debt=u(345);
nstep=min(cfg.forecast_hours,c.hours-first+1);
[action,~]=plan(c,first,nstep,state,currentStrategy);
g=action.gen; ch=action.ch; dis=action.dis; ds=action.direct_short;
direct=c.direct_fuel_demand_mtce(first,:)-ds;
fuelDelta=c.fuel_arrivals_mtce(first,:)-direct;
fuelDelta(1)=fuelDelta(1)-g(1)/(8141*c.fuel_efficiency.coal);
fuelDelta(3)=fuelDelta(3)-g(2)/(8141*c.fuel_efficiency.gas);
socDelta=zeros(1,2);
for j=1:2
    st=c.storage.(c.stores(j));
    socDelta(j)=st.eta_charge*ch(j)-dis(j)/st.eta_discharge;
end
y=[g,ch(1),dis(1),ch(2),dis(2),ds,action.out,action.repay, ...
    action.unserved,action.spill,fuelDelta,socDelta,action.out-action.repay,g(1:2)]';
v3_live_monitor('hourly',first-1, ...
    [state.fuel+fuelDelta,state.soc+socDelta,state.debt+action.out-action.repay]', ...
    [c.demand_gw(first);y]);
end

function [a,objective] = plan(c,first,nstep,state,strategy)
cfg=c.config;
names=["gen","ch","dis","soc","fuel","direct_short","out","repay","debt","unserved","spill"];
widths=[7,2,2,2,3,3,1,1,1,1,1]; ix=struct; offset=0;
for j=1:numel(names)
    width=widths(j); ix.(names(j))=reshape(offset+(1:nstep*width),width,nstep)';
    offset=offset+nstep*width;
end
cost=zeros(offset,1); lb=zeros(offset,1); ubound=inf(offset,1);
range=first:first+nstep-1;
available=c.available_gw(range,:); demand=c.demand_gw(range);
direct=c.direct_fuel_demand_mtce(range,:); arrivals=c.fuel_arrivals_mtce(range,:);
if strategy=="decentralized", local=localStorage(c,range,state); else, local=[]; end
ef=[94.6,73.3,56.1]*29.3076/1000;
for k=1:nstep
    urgency=1+1e-5*(nstep-k+1)/nstep;
    ubound(ix.gen(k,:))=available(k,:);
    cost(ix.gen(k,:))=cfg.generation_cost_cny_kwh;
    cost(ix.gen(k,1))=cost(ix.gen(k,1))+ef(1)/(8141*c.fuel_efficiency.coal)*cfg.carbon_price_cny_t;
    cost(ix.gen(k,2))=cost(ix.gen(k,2))+ef(3)/(8141*c.fuel_efficiency.gas)*cfg.carbon_price_cny_t;
    ubound(ix.direct_short(k,:))=direct(k,:);
    cost(ix.direct_short(k,:))=cfg.direct_shortage_cost_cny_kwh_th*8141*urgency ...
        -cfg.fuel_value_million_cny_mtce-ef*cfg.carbon_price_cny_t;
    for j=1:2
        st=c.storage.(c.stores(j));
        ubound(ix.soc(k,j))=st.energy_gwh;
        if k==nstep, cost(ix.soc(k,j))=-cfg.stored_electricity_value_million_cny_gwh; end
        ubound([ix.ch(k,j),ix.dis(k,j)])=st.power_gw;
        cost([ix.ch(k,j),ix.dis(k,j)])=cfg.storage_cycle_cost_cny_kwh;
        if ~isempty(local)
            lb(ix.ch(k,j))=local.ch(k,j); ubound(ix.ch(k,j))=local.ch(k,j);
            lb(ix.dis(k,j))=local.dis(k,j); ubound(ix.dis(k,j))=local.dis(k,j);
        end
    end
    ubound(ix.out(k))=cfg.flex_fraction*demand(k);
    ubound(ix.repay(k))=cfg.repay_fraction*demand(k);
    cost([ix.out(k),ix.repay(k)])=cfg.shift_cost_cny_kwh;
    cost(ix.unserved(k))=cfg.unserved_cost_cny_kwh*urgency;
    if mod(first+k-1,cfg.deadline_hours)==0 || first+k-1==c.hours
        ubound(ix.debt(k))=0;
    end
end
% Exactly seven equality rows per time: power, three fuels, two stores, debt.
eq=zeros(7*nstep,offset); rhs=zeros(7*nstep,1); ne=0;
% Power-service bound, shared storage power and four ramp inequalities.
ineq=zeros(7*nstep,offset); limit=zeros(7*nstep,1); ni=0;
for k=1:nstep
    ne=ne+1; eq(ne,ix.gen(k,:))=1; eq(ne,ix.dis(k,:))=1; eq(ne,ix.ch(k,:))=-1;
    eq(ne,ix.out(k))=1; eq(ne,ix.repay(k))=-1;
    eq(ne,ix.unserved(k))=1; eq(ne,ix.spill(k))=-1; rhs(ne)=demand(k);
    ni=ni+1; ineq(ni,[ix.unserved(k),ix.out(k)])=1; limit(ni)=demand(k);
    for j=1:3
        ne=ne+1; eq(ne,ix.fuel(k,j))=1; eq(ne,ix.direct_short(k,j))=-1;
        rhs(ne)=arrivals(k,j)-direct(k,j);
        if k==1, rhs(ne)=rhs(ne)+state.fuel(j); else, eq(ne,ix.fuel(k-1,j))=-1; end
        if j==1, eq(ne,ix.gen(k,1))=1/(8141*c.fuel_efficiency.coal); end
        if j==3, eq(ne,ix.gen(k,2))=1/(8141*c.fuel_efficiency.gas); end
    end
    for j=1:2
        st=c.storage.(c.stores(j)); ne=ne+1;
        eq(ne,ix.soc(k,j))=1; eq(ne,ix.ch(k,j))=-st.eta_charge;
        eq(ne,ix.dis(k,j))=1/st.eta_discharge;
        if k==1, rhs(ne)=state.soc(j); else, eq(ne,ix.soc(k-1,j))=-1; end
        ni=ni+1; ineq(ni,[ix.ch(k,j),ix.dis(k,j)])=1; limit(ni)=st.power_gw;
    end
    ne=ne+1; eq(ne,ix.debt(k))=1; eq(ne,ix.out(k))=-1; eq(ne,ix.repay(k))=1;
    if k==1, rhs(ne)=state.debt; else, eq(ne,ix.debt(k-1))=-1; end
    for j=1:2
        ramp=c.capacity_gw.(c.generators(j))*cfg.ramp_fraction(j);
        if k==1
            previous=state.thermal(j);
            previousAvailable=c.available_gw(max(1,first-1),j);
            drop=max(0,previousAvailable-available(k,j));
            ni=ni+1; ineq(ni,ix.gen(k,j))=1; limit(ni)=previous+ramp;
            ni=ni+1; ineq(ni,ix.gen(k,j))=-1; limit(ni)=max(ramp,drop)-previous;
        else
            drop=max(0,available(k-1,j)-available(k,j));
            ni=ni+1; ineq(ni,[ix.gen(k,j),ix.gen(k-1,j)])=[1,-1]; limit(ni)=ramp;
            ni=ni+1; ineq(ni,[ix.gen(k,j),ix.gen(k-1,j)])=[-1,1]; limit(ni)=max(ramp,drop);
        end
    end
end
options=optimoptions('linprog','Display','none');
[x,objective,exitflag,output]=linprog(cost,sparse(ineq),limit,sparse(eq),rhs,lb,ubound,options);
assert(exitflag>0,'V3:NativeLP','Native MPC hour %d failed: %s',first-1,output.message);
a=struct;
for j=1:numel(names), a.(names(j))=x(ix.(names(j))(1,:))'; end
end

function p = localStorage(c,range,state)
n=numel(range); net=c.demand_gw(range)-sum(c.available_gw(range,3:7),2);
p.ch=zeros(n,2); p.dis=zeros(n,2);
for j=1:2
    st=c.storage.(c.stores(j)); soc=state.soc(j);
    for k=1:n
        if net(k)<0
            p.ch(k,j)=min([-net(k),st.power_gw,max(0,st.energy_gwh-soc)/st.eta_charge]);
            soc=soc+p.ch(k,j)*st.eta_charge; net(k)=net(k)+p.ch(k,j);
        else
            p.dis(k,j)=min([net(k),st.power_gw,max(0,soc)*st.eta_discharge]);
            soc=soc-p.dis(k,j)/st.eta_discharge; net(k)=net(k)-p.dis(k,j);
        end
    end
end
end
