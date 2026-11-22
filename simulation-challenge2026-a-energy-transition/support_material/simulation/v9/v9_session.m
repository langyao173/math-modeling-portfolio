% AI-assisted: OpenAI Codex / GPT-6, OpenAI, used 2026-10-18; release date unverified.
function result=v9_session(action,varargin)
% One actual Simulink session. Actions-only LP with scalar native conservation. No replay is supplied to the solver.
% prepare(c,strategy) initializes; step(n) invokes Simulation.step; get reads;
% close stops and releases the native session. Prepared means zero steps.
persistent S
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
addpath(folder);addpath(fullfile(root,'simulation','v3'));addpath(fullfile(root,'simulation','v6'));
if isempty(S),S=emptySession();end
switch action
    case 'prepare'
        releaseSession();S=emptySession();c=varargin{1};strategy="native_coordinated_mpc";
        if numel(varargin)>1,strategy=string(varargin{2});end
        plain=erase(erase(strategy,"native_"),"_mpc");
        assert(any(plain==["coordinated","decentralized"]),'V9:Strategy','Unknown native strategy.');
        assert(license('test','Optimization_Toolbox') && exist('linprog','file')==2,'V9:License','Optimization Toolbox and linprog required.');
        assert(c.hours>=2 && size(c.available_gw,1)==c.hours && size(c.fuel_arrivals_mtce,1)==c.hours,'V9:Case','Case dimensions do not match its horizon.');
        assert(c.config.forecast_hours==24,'V9:Information','This native port requires the declared common 24h forecast.');
        assert(c.initial_debt_gwh==0,'V9:Debt','Native model starts with zero service debt.');
        c.generators=string(c.generators);c.fuels=string(c.fuels);c.stores=string(c.stores);
        c.stress_mask=logical(c.stress_mask(:));c.demand_gw=c.demand_gw(:);
        assert(all(isfinite([c.demand_gw;c.available_gw(:);c.fuel_arrivals_mtce(:);c.direct_fuel_demand_mtce(:)])),'V9:Case','Nonfinite external input.');
        modelPath=v9_build_model();[~,model]=fileparts(modelPath);
        v9_controller([],c,plain);v9_trace('reset');
        raw=[c.demand_gw,c.available_gw,c.fuel_arrivals_mtce,c.direct_fuel_demand_mtce];
        forecasts=zeros(c.hours,338);forecasts(:,1)=(0:c.hours-1)';forecasts(:,2)=forecasts(:,1);
        for k=1:c.hours
            rows=min(c.hours,k:k+23);forecasts(k,3:end)=reshape(raw(rows,:)',1,[]);
        end
        input=Simulink.SimulationInput(model);
        input=input.setVariable('v9_forecasts',forecasts);
        input=input.setVariable('v9_initial_fuel',c.initial_fuel_stocks_mtce);
        input=input.setVariable('v9_initial_soc',[c.storage.battery.initial_soc_gwh,c.storage.pumped.initial_soc_gwh]);
        input=input.setVariable('v9_initial_thermal',c.initial_thermal_gw);
        input=input.setVariable('v9_coal_conversion',1/(8141*c.fuel_efficiency.coal));
        input=input.setVariable('v9_gas_conversion',1/(8141*c.fuel_efficiency.gas));
        input=input.setVariable('v9_eta_charge',[c.storage.battery.eta_charge,c.storage.pumped.eta_charge]);
        input=input.setVariable('v9_eta_discharge',[c.storage.battery.eta_discharge,c.storage.pumped.eta_discharge]);
        input=input.setModelParameter('StopTime',num2str(c.hours-1),'EnablePacing','off');
        S=emptySession();S.case=c;S.strategy=strategy;S.plain=plain;S.model=model;S.model_path=modelPath;
        S.native=simulation(input);S.status='prepared';S.started=tic;
    case 'step'
        assert(any(strcmp(S.status,{'prepared','paused'})),'V9:Session','Prepare a session, or continue an unfinished paused session.');
        n=varargin{1};validateattributes(n,{'numeric'},{'scalar','integer','positive','finite'});
        S.status='running';
        % Native observer is downstream of the conservation ports.
        
        try
            ended=step(S.native,NumberOfSteps=n);
            evidence=v9_trace('get');S.trace=v6_session_trace(S.case,evidence);
            S.executed_steps=height(S.trace);S.runtime_seconds=toc(S.started);
            if ended || S.native.Status=="inactive"
                assert(S.executed_steps==S.case.hours,'V9:Incomplete','Native session ended before the declared horizon.');
                if S.native.Status=="inactive",nativeOutput=S.native.SimulationOutput;else,nativeOutput=stop(S.native);end
                S.status='complete';S.audit=v3_hourly_audit(S.case,S.trace);
                assert(S.audit.passed,'V9:Physics','Native independent physical audit failed.');
                S.summary=v3_hourly_summary(S.case,S.trace,S.plain,S.runtime_seconds);
                if isfield(S.case,'contract_premium_million_cny')
                    S.summary.contract_purchase_value_million_cny=S.case.contract_purchase_value_million_cny;
                    S.summary.contract_premium_million_cny=S.case.contract_premium_million_cny;
                    S.summary.resource_plus_holding_and_contract_premium_million_cny= ...
                        S.summary.resource_plus_holding_cost_million_cny+S.case.contract_premium_million_cny;
                end
                logged=squeeze(nativeOutput.get('v9_native_trace'));
                if size(logged,1)==26 && size(logged,2)==S.case.hours,logged=logged';end
                assert(isequal(size(logged),size(evidence.values)) && max(abs(logged-evidence.values),[],'all')<1e-7,'V9:Observation','Live evidence does not match final native logging.');
            else
                S.status='paused';
            end
        catch failure
            evidence=v9_trace('get');S.trace=v6_session_trace(S.case,evidence);S.executed_steps=height(S.trace);
            S.status='failed';S.error=getReport(failure,'extended','hyperlinks','off');
            if S.native.Status~="inactive",stop(S.native);end
            rethrow(failure);
        end
    case 'get'
        % Reading does not advance the simulation.
    case 'close'
        releaseSession();S=emptySession();
    otherwise
        error('V9:Session','Unknown session action.');
end
result=rmfield(S,{'native','started','plain'});
    function releaseSession()
        if ~isempty(S.native) && S.native.Status~="inactive",stop(S.native);end
        
    end
end

function S=emptySession()
S=struct('status','not_started','case',struct(),'strategy',"",'model',"",'model_path',"", ...
    'native',[],'started',[],'plain',"",'trace',table(),'executed_steps',0, ...
    'runtime_seconds',0,'audit',struct(),'summary',struct(),'error','');
end
