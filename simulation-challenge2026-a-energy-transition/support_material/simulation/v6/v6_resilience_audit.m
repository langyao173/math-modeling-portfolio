function receipt=v6_resilience_audit()
% Actually execute two full native 2160-hour trajectories from V6 knee state.
% AI-assisted, 2026-10-18. Python references are never controller inputs.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));addpath(fullfile(root,'simulation','v6'));addpath(fullfile(root,'simulation','v3'));
out=fullfile(root,'outputs','v6','native');if ~isfolder(out),mkdir(out);end
sources={'config/v6_resilience.json','src/v6_resilience.py','simulation/v6/v6_resilience_case.m', ...
    'simulation/v6/v6_resilience_audit.m','simulation/v6/v6_session.m','simulation/v6/v6_session_model.m', ...
    'simulation/v6/v6_session_observer.m','simulation/v6/v6_session_trace.m','simulation/v3/v3_hourly_mpc_step.m', ...
    'simulation/v3/v3_hourly_audit.m','outputs/v6/resilience/native_inputs.json'};
before=identities(root,sources);input=jsondecode(fileread(fullfile(root,'outputs','v6','resilience','native_inputs.json')));
records=struct();policies=["inherited","contract"];
for policy=policies
    c=v6_resilience_case([],policy,2160,"repeated");
    prepared=v6_session('prepare',c,"native_coordinated_mpc");assert(prepared.executed_steps==0 && strcmp(prepared.status,'prepared'));
    r=v6_session('step',2160);assert(strcmp(r.status,'complete') && r.executed_steps==2160 && r.audit.passed);
    traceFile=fullfile(out,"resilience_"+policy+".csv");writetable(r.trace,traceFile);
    raw=[c.demand_gw,c.available_gw,c.fuel_arrivals_mtce,c.direct_fuel_demand_mtce,c.emergency_arrivals_mtce];
    rawFile=fullfile(out,"resilience_inputs_"+policy+".csv");writematrix(raw,rawFile);
    record=struct('status',r.status,'hours',r.executed_steps,'strategy',r.strategy,'snapshot_key',input.default_snapshot_key, ...
        'supply_policy',policy,'physical_audit',r.audit,'summary',r.summary, ...
        'trace_file',relative(root,traceFile),'trace_sha256',sha(traceFile), ...
        'raw_input_file',relative(root,rawFile),'raw_input_sha256',sha(rawFile), ...
        'contract_purchase_value_million_cny',c.contract_purchase_value_million_cny, ...
        'contract_premium_million_cny',c.contract_premium_million_cny);
    records.(policy)=record;v6_session('close');fprintf('%s: native 2160-hour physical audit PASS\n',policy);
end
after=identities(root,sources);assert(isequal(before,after),'V6 sources changed during native cohort.');
receipt=struct('status','PASS_TWO_ACTUAL_NATIVE_CONTINUOUS_2160H_RUNS','native_runs',2,'hours_per_case',2160, ...
    'source_sha256_before',{before},'source_sha256_after',{after},'records',records, ...
    'model_file','outputs/v6/native/hourly_v6_session.slx','model_sha256',sha(fullfile(out,'hourly_v6_session.slx')), ...
    'scope','Actual MATLAB linprog and Simulink physical states. Common 24h perfect forecasts; one-way end-state stress experiment, not measured chronology or annual feedback.');
writeJson(fullfile(out,'resilience_native_audit.json'),receipt);
end
function rows=identities(root,files)
rows=cell(size(files));for j=1:numel(files),rows{j}=struct('path',files{j},'sha256',sha(fullfile(root,files{j})));end
end
function value=relative(root,path)
value=strrep(strrep(char(path),[root filesep],''),'\','/');
end
function value=sha(file)
fid=fopen(file,'rb');assert(fid>=0);bytes=fread(fid,Inf,'*uint8');fclose(fid);md=java.security.MessageDigest.getInstance('SHA-256');md.update(bytes);value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);fprintf(fid,'%s',jsonencode(value,'PrettyPrint',true));fclose(fid);
end
