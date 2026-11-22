% AI-assisted: OpenAI Codex / GPT-6, OpenAI, used 2026-10-18; release date unverified.
function receipt=v9_validate_contract()
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));addpath(fullfile(root,'simulation','v9'),fullfile(root,'simulation','v6'),fullfile(root,'simulation','v3'));
out=fullfile(root,'outputs','v9','native');if ~isfolder(out),mkdir(out);end
assert(strcmp(v9_session('get').status,'not_started'),'Close the V9 UI session before batch contract verification.');
c=v6_resilience_case([],"inherited",2160,"repeated");ref=readtable(fullfile(root,'outputs','v6','native','resilience_inherited.csv'));
receipt=struct('status','RUNNING','scope','145-hour native implementation equivalence and deterministic matched controls; unchanged scientific cohort','horizon',2160,'export_columns',49);
strategies={'native_coordinated_mpc','native_decentralized_mpc'};
for j=1:2
    v9_session('prepare',c,strategies{j});S=v9_session('step',145);T=S.trace;
    receipt.strategy(j)=struct('name',strategies{j},'steps',height(T),'ens_gwh',sum(T.unserved_gw),'shock_120_to_144_ens_gwh',sum(T.unserved_gw(T.hour>=120)),'hour138_unserved_gw',T.unserved_gw(T.hour==138),'native_power_balance_max_gw',max(abs(v9_trace('get').power_residual_gw)));
    if j==1,coordinated=T;receipt.all_49_columns_max_abs_difference=max(abs(T{:,:}-ref{1:145,:}),[],'all');assert(receipt.all_49_columns_max_abs_difference<1e-7);end
    writetable(T,fullfile(out,sprintf('native_%s_145h.csv',erase(erase(strategies{j},'native_'),'_mpc'))));v9_session('close');
end
receipt.observed_full_trajectory_ens_difference_gwh=receipt.strategy(2).ens_gwh-receipt.strategy(1).ens_gwh;
% Branch at h138 using precisely the same realized h137 physical state.
previous=coordinated(coordinated.hour==137,:);initialFuel=previous{1,{'coal_stock_mtce','oil_stock_mtce','gas_stock_mtce'}};initialSoc=previous{1,{'battery_soc_gwh','pumped_soc_gwh'}};initialThermal=previous{1,{'coal_ramp_state_gw','gas_ramp_state_gw'}};initialDebt=previous.shift_debt_gwh;
modelPath=v9_build_model();[~,model]=fileparts(modelPath);
forecast=zeros(2160,338);forecast(:,1)=(0:2159)';forecast(:,2)=forecast(:,1);raw=[c.demand_gw c.available_gw c.fuel_arrivals_mtce c.direct_fuel_demand_mtce];
for k=1:2160,rows=min(2160,k:k+23);forecast(k,3:end)=reshape(raw(rows,:)',1,[]);end
for j=1:2
    plain=erase(erase(string(strategies{j}),"native_"),"_mpc");v9_controller([],c,plain);v9_trace('reset');
    input=Simulink.SimulationInput(model);input=input.setVariable('v9_forecasts',forecast).setVariable('v9_initial_fuel',initialFuel).setVariable('v9_initial_soc',initialSoc).setVariable('v9_initial_thermal',initialThermal);
    input=input.setVariable('v9_coal_conversion',1/(8141*c.fuel_efficiency.coal)).setVariable('v9_gas_conversion',1/(8141*c.fuel_efficiency.gas)).setVariable('v9_eta_charge',[c.storage.battery.eta_charge c.storage.pumped.eta_charge]).setVariable('v9_eta_discharge',[c.storage.battery.eta_discharge c.storage.pumped.eta_discharge]);
    input=input.setBlockParameter([model '/Physical_Resource_Network/Shift_Debt/Previous_Debt_GWh'],'InitialCondition',num2str(initialDebt,17));input=input.setModelParameter('StartTime','138','StopTime','138');sim(input);
    e=v9_trace('get');assert(isequal(e.hours,138) && size(e.values,1)==1);r=e.values;
    receipt.same_state_hour138(j)=struct('strategy',strategies{j},'initial_fuel_mtce',initialFuel,'initial_soc_gwh',initialSoc,'initial_thermal_gw',initialThermal,'initial_debt_gwh',initialDebt,'unserved_gw',r(17),'battery_discharge_gw',r(9),'pumped_discharge_gw',r(11),'end_states',r(19:26),'power_residual_gw',e.power_residual_gw,'lp',v9_controller('get'));
end
receipt.same_state_ens_difference_gw=receipt.same_state_hour138(2).unserved_gw-receipt.same_state_hour138(1).unserved_gw;
receipt.same_state_forecast='Identical 8 pre-action states and next-24h exogenous inputs at h138; only strategy differs. Separate from full-trajectory effect.';
% The full horizon exercises near-zero fuel states and native completion logging.
for j=1:2
    policies={'inherited','contract'};policy=policies{j};fullCase=v6_resilience_case([],string(policy),2160,"repeated");
    v9_session('prepare',fullCase,'native_coordinated_mpc');F=v9_session('step',2160);frozen=readtable(fullfile(root,'outputs','v6','native',['resilience_' policy '.csv']));
    difference=max(abs(F.trace{:,:}-frozen{:,:}),[],'all');assert(F.audit.passed && F.executed_steps==2160 && strcmp(F.status,'complete') && difference<1e-7);
    receipt.full_horizon(j)=struct('supply_policy',policy,'steps',2160,'all_49_columns_max_abs_difference',difference,'native_audit',F.audit,'ens_gwh',sum(F.trace.unserved_gw),'minimum_fuel_stocks_mtce',min(F.trace{:,{'coal_stock_mtce','oil_stock_mtce','gas_stock_mtce'}},[],1),'native_logged_trace_compared',true);
    writetable(F.trace,fullfile(out,['native_' policy '_coordinated_2160h.csv']));v9_session('close');
    fid=fopen(fullfile(out,'full_horizon_progress.json'),'wb');fwrite(fid,unicode2native(jsonencode(receipt.full_horizon,'PrettyPrint',true),'UTF-8'),'uint8');fclose(fid);
end
receipt.status='PASS_FULL_2160H_NATIVE_EQUIVALENCE_AND_145H_STRATEGY_PAIRS';
files={'simulation/v3/v3_hourly_mpc_step.m','simulation/v9/v9_controller.m','simulation/v9/v9_build_model.m','simulation/v9/V9_Whitebox_Resource_Network.slx','simulation/v9/v9_session.m','simulation/v9/v9_trace.m','simulation/v9/v9_validate_contract.m'};
for j=1:numel(files),receipt.files(j)=struct('path',files{j},'sha256',hash(fullfile(root,files{j})));end
receipt.utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ssXXX'));fid=fopen(fullfile(out,'model_contract.json'),'wb');fwrite(fid,unicode2native(jsonencode(receipt,'PrettyPrint',true),'UTF-8'),'uint8');fclose(fid);
end
function value=hash(path)
fid=fopen(path,'rb');bytes=fread(fid,Inf,'*uint8');fclose(fid);d=java.security.MessageDigest.getInstance('SHA-256');d.update(bytes);value=lower(reshape(dec2hex(typecast(d.digest(),'uint8'),2).',1,[]));
end
