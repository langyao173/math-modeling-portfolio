function report=v6_annual_audit()
% Actual V6 closed-loop native runs; Python future traces are audit-only data.
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));addpath(folder);addpath(fullfile(root,'simulation','v3'),'-end');
out=fullfile(root,'outputs','v6','native');
input=load(fullfile(out,'annual_native_inputs.mat'));
reference=load(fullfile(out,'annual_python_reference.mat'));
manifest=jsondecode(fileread(fullfile(out,'annual_inputs_manifest.json')));
names=cellstr(string(input.audit_case_names(:)));
files={'simulation/v6/v6_annual_step.m','simulation/v6/v6_annual_run.m','simulation/v6/v6_build_annual_model.m','simulation/v6/v6_annual_audit.m', ...
    'simulation/v3/v3_annual_expectations.m','simulation/v3/v3_annual_forecast_level.m','simulation/v3/v3_execute.m','simulation/v3/v3_live_monitor.m', ...
    'src/v6_native_inputs.py','src/v6_annual.py','src/v3_annual.py','src/v3_annual_calibrated.py','src/v3_calibration_candidate.py', ...
    'config/v3_transition.json','data/v6_fuel_import_channels.csv','data/annual_china.csv','data/ember_power_china.csv','data/energy_production_china.csv','data/fuel_supply_trade.csv','data/supplement_carriers_2024.csv', ...
    'outputs/v6/native/annual_native_inputs.mat','outputs/v6/native/annual_python_reference.mat','outputs/v6/native/annual_inputs_manifest.json','outputs/v6/native/annual_cases.json', ...
    'outputs/v6/planning/selected_cases.csv','outputs/v6/planning/planning_audit.json'};
before=fileHashes(root,files);reports=struct();outputs=struct();
report=struct('status','RUNNING','source_input_hashes_before',{before},'cases',reports, ...
    'started_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')));
writeJson(fullfile(out,'annual_native_audit.json'),report);
try
    v6_build_annual_model(true);
    for i=1:numel(names)
        name=names{i};fprintf('V6 native annual case %d/%d: %s\n',i,numel(names),name);
        result=v6_annual_run(name,struct());
        expected=readtable(fullfile(out,['annual_python_' name '.csv']),'TextType','string');
        expectedActions=readtable(fullfile(out,['annual_python_' name '_actions.csv']),'TextType','string');
        [traceRelative,traceAbsolute,traceCount]=compareTables(expected,result.trace);
        [actionRelative,actionAbsolute,actionCount]=compareTables(expectedActions,result.actions);
        expectedState=reference.references.(name).state;expectedMetrics=reference.references.(name).metrics;
        stateRelative=max(abs(expectedState-result.state)./max(abs(expectedState),1),[],'all');
        metricsRelative=max(abs(expectedMetrics-result.raw_metrics)./max(abs(expectedMetrics),1),[],'all');
        stateAbsolute=max(abs(expectedState-result.state),[],'all');
        metricsAbsolute=max(abs(expectedMetrics-result.raw_metrics),[],'all');
        largestRelative=max([traceRelative actionRelative stateRelative metricsRelative]);
        assert(height(result.trace)==36 && height(result.actions)==36*12,'V6:Incomplete','Expected 36 annual decisions and 432 investor actions');
        assert(largestRelative<1e-8,'V6:NativeParity','Native V6 state/action differs from Python reference');
        assert(result.metrics.max_balance_error<1e-7,'V6:Balance','Native V6 physical balance failed');
        assert(all(result.trace.investment_spend_million_cny<=result.trace.investment_budget_million_cny+1e-5),'V6:Budget','Native V6 shared capital budget exceeded');
        for fuel={'coal','oil','gas'}
            assert(all(result.trace.([fuel{1} '_stock_mtce'])>=-1e-8),'V6:Stock','Negative native fuel stock');
        end
        grouped=groupsummary(result.actions,'year','sum','capital_spend_million_cny');
        spendError=max(abs(grouped.sum_capital_spend_million_cny-result.trace.investment_spend_million_cny));
        assert(spendError<1e-5,'V6:Actions','Native investor expenditures do not reconstruct actual annual spend');
        tracePath=fullfile(out,['annual_native_' name '.csv']);actionsPath=fullfile(out,['annual_native_' name '_actions.csv']);
        statesPath=fullfile(out,['annual_native_' name '_states.csv']);metricsPath=fullfile(out,['annual_native_' name '_raw_metrics.csv']);
        writetable(result.trace,tracePath);writetable(result.actions,actionsPath);writematrix(result.state,statesPath);writematrix(result.raw_metrics,metricsPath);
        reports.(name)=struct('status','PASS','annual_steps',36,'investor_actions',432,'trace_values_compared',traceCount, ...
            'action_values_compared',actionCount,'state_values_compared',numel(expectedState),'raw_metric_values_compared',numel(expectedMetrics), ...
            'max_relative_error',largestRelative,'max_absolute_error',max([traceAbsolute actionAbsolute stateAbsolute metricsAbsolute]), ...
            'max_balance_error',result.metrics.max_balance_error,'action_spend_error_million_cny',spendError,'metrics',result.metrics);
        outputs.(name)=struct('trace_sha256',shaFile(tracePath),'actions_sha256',shaFile(actionsPath),'state_sha256',shaFile(statesPath),'metrics_sha256',shaFile(metricsPath));
        report.cases=reports;report.output_hashes=outputs;writeJson(fullfile(out,'annual_native_audit.json'),report);
        clear result
    end
    after=fileHashes(root,files);
    assert(isequal(before,after),'V6:SourceDrift','Source or input changed during native V6 execution');
    report.status='PASS_ALL_DECLARED_ACTUAL_NATIVE_V6_ANNUAL_RUNS';report.native_runs=numel(names);report.cases=reports;
    report.source_input_hashes_after=after;report.output_hashes=outputs;
    report.native_state_count=217;report.output_width=349;report.new_order_minimum_lag_years=1;
    report.role_ids=manifest.role_ids;
    report.model_path='simulation/v6/V6_Energy_Transition_ABM.slx';report.model_sha256=shaFile(fullfile(folder,'V6_Energy_Transition_ABM.slx'));
    report.scope='All declared audit cases execute native annual state transitions, including four scenarios, origin-anchored and legacy profit rules, mechanism interventions and role-selected plans. Agreement checks implementation, not real-world forecast accuracy.';
    report.input_boundary='Only observed initial state, fixed calibration and time-constant mechanism/decision policies enter native simulation. Separate future reference file is loaded by audit only.';
    report.completed_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
    writeJson(fullfile(out,'annual_native_audit.json'),report);disp(report.status);
catch exception
    report.status='FAILED_PARTIAL_NATIVE_V6_NOT_FOR_SUCCESS_CLAIM';report.cases=reports;
    report.error=getReport(exception,'extended','hyperlinks','off');writeJson(fullfile(out,'annual_native_audit.json'),report);
    rethrow(exception);
end
end

function [relative,absolute,count]=compareTables(expected,actual)
assert(height(expected)==height(actual),'V6:Rows','Comparison rows mismatch');
assert(all(ismember(expected.Properties.VariableNames,actual.Properties.VariableNames)),'V6:Columns','Native output omits a declared Python trace/action field');
common=expected.Properties.VariableNames;relative=0;absolute=0;count=0;
for j=1:numel(common)
    a=expected.(common{j});b=actual.(common{j});
    % pandas writes True/False; MATLAB CSV inference retains their text case.
    if islogical(b) && isstring(a)
        assert(all(ismember(lower(a),["true","false"])),'V6:Boolean','Expected strict boolean CSV tokens');
        a=strcmpi(a,"true");
    end
    if (isnumeric(a) || islogical(a)) && (isnumeric(b) || islogical(b))
        a=double(a);b=double(b);
        assert(all(isfinite(a)|isnan(a)) && all(isfinite(b)|isnan(b)),'V6:Finite','Infinite output values');
        assert(isequal(isnan(a),isnan(b)),'V6:Finite','Comparison NaN pattern mismatch');finite=isfinite(a)&isfinite(b);
        if any(finite)
            relative=max(relative,max(abs(a(finite)-b(finite))./max(abs(a(finite)),1)));
            absolute=max(absolute,max(abs(a(finite)-b(finite))));count=count+sum(finite);
        end
    else
        assert(isequal(string(a),string(b)),'V6:Labels','Native and Python metadata differ');
    end
end
end

function result=fileHashes(root,paths)
result=cell(size(paths));
for j=1:numel(paths),result{j}=struct('path',paths{j},'sha256',shaFile(fullfile(root,paths{j})));end
end

function digest=shaFile(path)
fid=fopen(path,'rb');assert(fid~=-1,'V6:Hash','Cannot open source or artifact for hashing');
bytes=fread(fid,Inf,'*uint8');fclose(fid);md=java.security.MessageDigest.getInstance('SHA-256');md.update(bytes);
digest=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end

function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');fprintf(fid,'%s',jsonencode(value,'PrettyPrint',true));fclose(fid);
end
