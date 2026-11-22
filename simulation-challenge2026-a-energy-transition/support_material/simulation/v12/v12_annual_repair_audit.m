function report=v12_annual_repair_audit()
% Execute frozen repair_005 in the unchanged V6 native annual model.
% Reference states/actions are loaded only after actual simulation returns.
% AI-assisted: OpenAI Codex, 2026-10-19.
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
out=fullfile(root,'outputs','v12','native');receiptPath=fullfile(out,'annual_native_audit.json');
assert(~isfile(receiptPath),'V12:ExistingReceipt','Keep the existing native execution receipt unchanged.');
manifestPath=fullfile(out,'preparation_manifest.json');
manifest=jsondecode(fileread(manifestPath));
assert(strcmp(manifest.status,'PASS_PYTHON_REFERENCE_PREPARED_NATIVE_EXECUTION_PENDING'),'V12:Preparation','Prepared fixed inputs required.');
assert(strcmp(manifest.candidate_id,'repair_005'),'V12:Candidate','Only frozen repair_005 is authorized here.');
paths=[manifest.source_input_hashes_after(:);manifest.prepared_artifacts(:)];
for i=1:numel(paths)
    assert(strcmp(shaFile(fullfile(root,paths(i).path)),paths(i).sha256),'V12:PreparedHash','Prepared source/input hash changed: %s',paths(i).path);
end
before=fileHashes(root,{paths.path});
model='V6_Energy_Transition_ABM';modelPath=fullfile(root,'simulation','v6',[model '.slx']);
assert(isfile(modelPath),'V12:Model','The frozen existing V6 model is required; this entry never rebuilds it.');
if bdIsLoaded(model)
    assert(strcmp(get_param(model,'SimulationStatus'),'stopped') && strcmp(get_param(model,'Dirty'),'off'),'V12:SharedModel','Wait for the existing model and preserve unsaved edits.');
    assert(strcmpi(get_param(model,'FileName'),modelPath),'V12:SharedModel','A different checkout owns the loaded V6 model.');
end
controls=load(fullfile(out,'native_controls.mat'));
expectedFields={'name';'investment_priority';'budget_multipliers_by_period';'carbon_price_drift_cny_t'};
assert(isequal(sort(fieldnames(controls.factors)),sort(expectedFields)),'V12:Boundary','Native inputs may contain constant control policies only.');
assert(strcmp(controls.factors.name,'repair_005'),'V12:Candidate','Native control identity mismatch.');
addpath(fullfile(root,'simulation','v6'));addpath(fullfile(root,'simulation','v3'),'-end');
report=struct('status','RUNNING_NATIVE_EXECUTION','candidate_id','repair_005', ...
    'started_utc',utcNow(),'matlab_version',version,'simulink',ver('simulink'), ...
    'source_input_hashes_before',before,'preparation_manifest_sha256',shaFile(manifestPath), ...
    'native_input_boundary','Original V6 observed initial state/fixed calibration plus constant factors only; reference values are read after native simulation');
writeJson(receiptPath,report);
try
    % v6_annual_run uses the existing model without a force-rebuild request.
    result=v6_annual_run('baseline',controls.factors);
    report.native_execution_returned_utc=utcNow();
    % Preserve actual native output before making any reference comparison.
    writetable(result.trace,fullfile(out,'native_trace.csv'));
    writetable(result.actions,fullfile(out,'native_actions.csv'));
    writematrix(result.state,fullfile(out,'native_states.csv'));
    writematrix(result.raw_metrics,fullfile(out,'native_raw_metrics.csv'));
    writeJson(fullfile(out,'native_executed_inputs.json'),result.executed_inputs);
    assert(isequal(size(result.state),[36 217]) && isequal(size(result.raw_metrics),[36 132]) && height(result.actions)==432,'V12:Incomplete','All 36 years and 432 investor actions are required.');
    assert(all(isfinite(result.state),'all') && all(isfinite(result.raw_metrics),'all'),'V12:Finite','Native states and raw metrics must be finite.');

    % First numerical reference reads occur here, after genuine native execution.
    reference=load(fullfile(out,'python_reference.mat'));
    expected=readtable(fullfile(out,'python_trace.csv'),'TextType','string');
    expectedActions=readtable(fullfile(out,'python_actions.csv'),'TextType','string');
    expectedSummary=jsondecode(fileread(fullfile(out,'python_summary.json')));
    [traceRelative,traceAbsolute,traceCount]=compareTables(expected,result.trace);
    [actionRelative,actionAbsolute,actionCount]=compareTables(expectedActions,result.actions);
    stateRelative=max(abs(reference.state-result.state)./max(abs(reference.state),1),[],'all');
    metricRelative=max(abs(reference.raw_metrics-result.raw_metrics)./max(abs(reference.raw_metrics),1),[],'all');
    stateAbsolute=max(abs(reference.state-result.state),[],'all');
    metricAbsolute=max(abs(reference.raw_metrics-result.raw_metrics),[],'all');
    assert(max([traceRelative actionRelative stateRelative metricRelative])<1e-8,'V12:Parity','Full native state/action/reference agreement failed.');

    original=load(fullfile(root,'outputs','v6','native','annual_native_inputs.mat'));
    p=original.parameters;s=result.executed_inputs.scenario;e=p.engineering;
    initial=double(original.initial_state(:)');initial(213)=s.carbon_price_start_cny_t;
    assert(max(abs(initial-reference.initial_state(:)'))<1e-10,'V12:Initial','Native and Python observed initial states differ.');
    previous=[initial;result.state(1:end-1,:)];
    oldAssets=previous(:,6:17);retired=oldAssets.*double(e.retirement_fraction(:)');
    m=result.raw_metrics;states=result.state;
    assetError=max(abs(states(:,6:17)-(oldAssets-retired+m(:,55:66))),[],'all');
    retirementError=max(abs(m(:,67:78)-retired),[],'all');
    queueError=0;commissionError=0;buildCapExcess=0;
    delayMultiplier=1;if isfield(s,'construction_delay_multiplier'),delayMultiplier=s.construction_delay_multiplier;end
    delays=ceil(double(e.construction_years(:))*delayMultiplier);
    for i=1:36
        priorQueue=reshape(previous(i,21:200),15,12)';
        commissionError=max(commissionError,max(abs(priorQueue(:,1)'-m(i,55:66))));
        nextQueue=[priorQueue(:,2:15),zeros(12,1)];
        for j=1:12,nextQueue(j,delays(j))=nextQueue(j,delays(j))+m(i,42+j);end
        actualQueue=reshape(states(i,21:200),15,12)';
        queueError=max(queueError,max(abs(nextQueue-actualQueue),[],'all'));
        elapsed=double(result.trace.year(i))-double(p.base_year);base=double(p.base_orders(:));
        future=p.reference_capacity0(:)+p.reference_net_increment(:).*(elapsed+delays(1:7));
        prior=p.reference_capacity0(:)+p.reference_net_increment(:).*(elapsed+delays(1:7)-1);
        base(1:7)=max(future-(1-double(e.retirement_fraction(1:7)')).*prior,0);
        base(10:12)=max(states(i,15:17)'.*(expm1(p.production_growth(:))+double(e.retirement_fraction(10:12)')),0);
        limit=max(oldAssets(i,:)'.*[.15;.20;.15;.08;.40;.50;.20;.50;.15;.15;.10;.20],base*1.5);
        buildCapExcess=max(buildCapExcess,max(m(i,43:54)'-limit));
    end
    assert(max([assetError retirementError queueError commissionError buildCapExcess])<1e-7,'V12:Construction','Construction/retirement/order-rate accounting failed.');
    fuelBalance=max(abs(m(:,31:33)+m(:,34:36)-m(:,40:42)-m(:,25:27)-m(:,28:30)),[],'all');
    stockChangeError=max(abs(states(:,18:20)-previous(:,18:20)-m(:,40:42)),[],'all');
    powerBalance=max(abs(result.trace.electricity_generation_twh+result.trace.electricity_unserved_twh-result.trace.electricity_demand_twh));
    domesticBound=states(:,15:17).*p.domestic_retention_fraction(:)'*s.supply_multiplier;
    importBound=states(:,214:216)*s.import_multiplier;
    for i=1:36
        if any(result.trace.year(i)==s.annual_shock_years),importBound(i,:)=importBound(i,:)*(1-s.import_shock_fraction);end
    end
    domesticExcess=max(m(:,31:33)-domesticBound,[],'all');importExcess=max(m(:,34:36)-importBound,[],'all');
    powerFuelError=max(abs(m(:,28:30)-[m(:,15)/(.4*(29.3076/3.6)),zeros(36,1),m(:,16)/(.55*(29.3076/3.6))]),[],'all');
    biomassExcess=max(result.trace.biomass_generation_twh-result.trace.biomass_generation_budget_twh_per_year);
    assert(max([fuelBalance stockChangeError powerBalance domesticExcess importExcess powerFuelError biomassExcess])<1e-7,'V12:Physical','Native fuel/power/feedstock accounting failed.');
    assert(all(states(:,18:20)>=-1e-8,'all'),'V12:Stocks','Native stocks must remain nonnegative.');
    budgetExcess=max(result.trace.investment_spend_million_cny-result.trace.investment_budget_million_cny);
    grouped=groupsummary(result.actions,'year','sum','capital_spend_million_cny');
    actionSpendError=max(abs(grouped.sum_capital_spend_million_cny-result.trace.investment_spend_million_cny));
    rawSpendError=max(abs(sum(m(:,43:54).*m(:,91:102),2)-result.trace.investment_spend_million_cny));
    assert(max([budgetExcess actionSpendError rawSpendError])<1e-5,'V12:Capital','Native paid investment/budget accounting failed.');
    discount=(1.04).^(result.actions.year-2024);
    capital=sum(result.actions.capital_spend_million_cny./discount)/1e6;
    paidCapital=sum(result.actions.capital_spend_million_cny)/1e6;
    net=result.trace.net_co2_mt(end);
    assert(abs(capital-expectedSummary.discounted_capital_trillion_cny)<1e-7 && abs(net-expectedSummary.net_emissions_2060_mt)<1e-7,'V12:Nominal','Frozen nominal capital/net emissions were not reproduced.');

    after=fileHashes(root,{paths.path});
    assert(isequal(before,after),'V12:FrozenDrift','A source, reference, old MAT/model or frozen V11 file changed during execution.');
    outputNames={'native_trace.csv','native_actions.csv','native_states.csv','native_raw_metrics.csv','native_executed_inputs.json'};
    outputHashes=struct();for i=1:numel(outputNames),outputHashes.(matlab.lang.makeValidName(outputNames{i}))=shaFile(fullfile(out,outputNames{i}));end
    report.status='PASS_ACTUAL_NATIVE_REPAIR_005_217_STATE_AND_PHYSICAL_AUDIT';
    report.native_runs=1;report.annual_steps=36;report.investor_actions=432;
    report.state_values_compared=numel(reference.state);report.raw_metric_values_compared=numel(reference.raw_metrics);
    report.trace_values_compared=traceCount;report.action_values_compared=actionCount;
    report.max_relative_error=max([traceRelative actionRelative stateRelative metricRelative]);
    report.max_absolute_error=max([traceAbsolute actionAbsolute stateAbsolute metricAbsolute]);
    report.construction=struct('asset_max_error',assetError,'retirement_max_error',retirementError,'queue_max_error',queueError,'commission_max_error',commissionError,'build_cap_excess',max(0,buildCapExcess));
    report.physical=struct('fuel_balance_max_mtce',fuelBalance,'stock_change_max_mtce',stockChangeError,'power_balance_max_twh',powerBalance, ...
        'domestic_supply_cap_excess_mtce',max(0,domesticExcess),'import_supply_cap_excess_mtce',max(0,importExcess),'power_fuel_error_mtce',powerFuelError,'biomass_budget_excess_twh',max(0,biomassExcess),'minimum_fuel_stock_mtce',min(states(:,18:20),[],'all'));
    report.capital=struct('budget_excess_million_cny',max(0,budgetExcess),'action_spend_error_million_cny',actionSpendError,'raw_spend_error_million_cny',rawSpendError, ...
        'discounted_capital_trillion_cny',capital,'paid_capital_trillion_cny',paidCapital);
    report.nominal_net_emissions_2060_mt=net;report.executed_inputs=result.executed_inputs;
    report.source_input_hashes_after=after;report.output_hashes=outputHashes;
    report.model_path='simulation/v6/V6_Energy_Transition_ABM.slx';report.model_sha256=shaFile(modelPath);
    report.scope='One frozen nominal repair_005 native annual execution. Implementation parity and physical accounting only; no new robustness search or independent-design retest.';
    report.completed_utc=utcNow();writeJson(receiptPath,report);disp(report.status);
catch exception
    report.status='FAILED_NATIVE_REPAIR_AUDIT_ACTUAL_OUTPUTS_RETAINED';report.error=getReport(exception,'extended','hyperlinks','off');
    writeJson(receiptPath,report);rethrow(exception);
end
end

function [relative,absolute,count]=compareTables(expected,actual)
assert(height(expected)==height(actual),'V12:Rows','Comparison row count differs.');
assert(all(ismember(expected.Properties.VariableNames,actual.Properties.VariableNames)),'V12:Columns','Native trace/action omits a Python reference field.');
relative=0;absolute=0;count=0;
for j=1:numel(expected.Properties.VariableNames)
    name=expected.Properties.VariableNames{j};a=expected.(name);b=actual.(name);
    if islogical(b) && isstring(a),assert(all(ismember(lower(a),["true","false"])));a=strcmpi(a,"true");end
    if (isnumeric(a)||islogical(a)) && (isnumeric(b)||islogical(b))
        a=double(a);b=double(b);assert(all(isfinite(a)|isnan(a)) && all(isfinite(b)|isnan(b)) && isequal(isnan(a),isnan(b)),'V12:Finite','Numeric/NaN patterns differ.');
        finite=isfinite(a)&isfinite(b);
        if any(finite),relative=max(relative,max(abs(a(finite)-b(finite))./max(abs(a(finite)),1)));absolute=max(absolute,max(abs(a(finite)-b(finite))));count=count+sum(finite);end
    else
        assert(isequal(string(a),string(b)),'V12:Labels','Native/Python trace labels differ.');
    end
end
end

function values=fileHashes(root,paths)
values=repmat(struct('path','','sha256',''),numel(paths),1);
for i=1:numel(paths),values(i).path=paths{i};values(i).sha256=shaFile(fullfile(root,paths{i}));end
end

function value=shaFile(path)
fid=fopen(path,'rb');assert(fid>=0,'V12:Hash','Cannot read %s',path);bytes=fread(fid,Inf,'*uint8');fclose(fid);
md=java.security.MessageDigest.getInstance('SHA-256');md.update(bytes);value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end

function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0,'V12:Write','Cannot write audit output.');fprintf(fid,'%s',jsonencode(value,'PrettyPrint',true));fclose(fid);
end

function value=utcNow()
value=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
end
