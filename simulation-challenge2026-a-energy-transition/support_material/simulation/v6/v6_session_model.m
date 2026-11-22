% AI-assisted V6 native session model; frozen V3 physical controller retained.
function modelPath = v6_session_model(forceRebuild)
% Native forecast/control/physical-state loop. No Python action replay input.
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder));
if nargin<1,forceRebuild=false;end
output=fullfile(root,'outputs','v6','native'); if ~isfolder(output), mkdir(output); end
model='hourly_v6_session'; modelPath=fullfile(output,[model '.slx']);
if bdIsLoaded(model)
    assert(~strcmp(get_param(model,'Dirty'),'on'),'Save existing modified native hourly model before rebuilding.');
    if ~forceRebuild,return,end
    assert(strcmp(get_param(model,'SimulationStatus'),'stopped'),'V6:ActiveModel','Pause preserves a live native session; rebuilding requires ending that session first.');
    close_system(model,0);
elseif isfile(modelPath) && ~forceRebuild
    load_system(modelPath);return;
end
new_system(model);
set_param(model,'Solver','FixedStepDiscrete','FixedStep','1','StartTime','0', ...
    'StopTime','719','ReturnWorkspaceOutputs','on','SaveTime','on', ...
    'TimeSaveName','v3hour_time','SaveOutput','off','SaveFormat','Array');
set_param(model,'Location',[80,80,1440,880]);
add_block('simulink/Sources/From Workspace',[model '/Demand_Weather_Fuel_Arrival_Forecasts'], ...
    'VariableName','v3hour_forecasts','Interpolate','off', ...
    'OutputAfterFinalValue','Holding final value','Position',[35,110,255,165]);
add_block('simulink/Signal Routing/Mux',[model '/Forecast_and_Physical_State'], ...
    'Inputs','5','Position',[325,125,330,465]);
load_system('simulink');
interpreted=find_system('simulink','SearchDepth',2,'Name','Interpreted MATLAB Function');
assert(numel(interpreted)==1,'Installed interpreted MATLAB block must be unambiguous.');
add_block(interpreted{1},[model '/Native_24h_MPC_Market'], ...
    'MATLABFcn','v3_hourly_mpc_step(u)','OutputDimensions','26', ...
    'Position',[390,220,600,290]);
add_block('simulink/Signal Routing/Demux',[model '/Controls_and_State_Increments'], ...
    'Outputs','[18 3 2 1 2]','Position',[645,170,650,435]);
integrator(model,'Coal_Oil_Gas_Stocks','v3hour_initial_fuel',[725,75,940,140]);
integrator(model,'Battery_Pumped_SOC','v3hour_initial_soc',[725,180,940,245]);
integrator(model,'Flexible_Load_Service_Debt','0',[725,285,940,350]);
memory(model,'Coal_Gas_Ramp_State','v3hour_initial_thermal',[725,390,940,455]);
add_block('simulink/Signal Routing/Mux',[model '/Export_Realized_Actions_and_End_States'], ...
    'Inputs','5','Position',[1050,115,1055,430]);
add_block('simulink/Sinks/To Workspace',[model '/Native_Hourly_Trace'], ...
    'VariableName','v3hour_trace','SaveFormat','Array','MaxDataPoints','inf', ...
    'Position',[1130,300,1300,345]);
add_block('simulink/Signal Routing/Selector',[model '/Power_Security_Channels'], ...
    'InputPortWidth','18','Indices','[1 2 17 18]','Position',[1045,25,1130,65]);
add_block('simulink/Sinks/Scope',[model '/State_and_Security_Scope'], ...
    'NumInputPorts','4','Position',[1190,70,1315,150]);
add_line(model,'Demand_Weather_Fuel_Arrival_Forecasts/1','Forecast_and_Physical_State/1','autorouting','on');
add_line(model,'Coal_Oil_Gas_Stocks/1','Forecast_and_Physical_State/2','autorouting','on');
add_line(model,'Coal_Gas_Ramp_State/1','Forecast_and_Physical_State/3','autorouting','on');
add_line(model,'Battery_Pumped_SOC/1','Forecast_and_Physical_State/4','autorouting','on');
add_line(model,'Flexible_Load_Service_Debt/1','Forecast_and_Physical_State/5','autorouting','on');
add_line(model,'Forecast_and_Physical_State/1','Native_24h_MPC_Market/1','autorouting','on');
add_line(model,'Native_24h_MPC_Market/1','Controls_and_State_Increments/1','autorouting','on');
add_line(model,'Controls_and_State_Increments/2','Coal_Oil_Gas_Stocks/1','autorouting','on');
add_line(model,'Controls_and_State_Increments/3','Battery_Pumped_SOC/1','autorouting','on');
add_line(model,'Controls_and_State_Increments/4','Flexible_Load_Service_Debt/1','autorouting','on');
add_line(model,'Controls_and_State_Increments/5','Coal_Gas_Ramp_State/1','autorouting','on');
add_line(model,'Controls_and_State_Increments/1','Export_Realized_Actions_and_End_States/1','autorouting','on');
add_line(model,'Coal_Oil_Gas_Stocks/2','Export_Realized_Actions_and_End_States/2','autorouting','on');
add_line(model,'Coal_Gas_Ramp_State/2','Export_Realized_Actions_and_End_States/3','autorouting','on');
add_line(model,'Battery_Pumped_SOC/2','Export_Realized_Actions_and_End_States/4','autorouting','on');
add_line(model,'Flexible_Load_Service_Debt/2','Export_Realized_Actions_and_End_States/5','autorouting','on');
add_line(model,'Export_Realized_Actions_and_End_States/1','Native_Hourly_Trace/1','autorouting','on');
add_line(model,'Controls_and_State_Increments/1','Power_Security_Channels/1','autorouting','on');
add_line(model,'Power_Security_Channels/1','State_and_Security_Scope/1','autorouting','on');
add_line(model,'Coal_Oil_Gas_Stocks/2','State_and_Security_Scope/2','autorouting','on');
add_line(model,'Battery_Pumped_SOC/2','State_and_Security_Scope/3','autorouting','on');
add_line(model,'Flexible_Load_Service_Debt/2','State_and_Security_Scope/4','autorouting','on');
note=Simulink.Annotation(model,sprintf(['NATIVE MULTI-ENERGY DYNAMIC SIMULATION | 1 step = 1 hour\n' ...
    '14 raw exogenous channels x 24h forecast -> fresh MATLAB linprog -> FIRST ACTION ONLY.\n' ...
    'Physical states: three fuel stocks, two thermal ramp states, two SOCs, flexible service debt.\n' ...
    'Fuel conversion and storage losses are enforced; direct fuel services compete with power.\n' ...
    'Designed stress experiment, aggregate bus, continuous plants, no historical hourly claim.']));
note.Position=[35,545,1280,665];
set_param(model,'Description','AI-assisted reference model, 2026-10-18. Native state-feedback 24h MPC.');
set_param(model,'Creator','Simulation Team','ModifiedByFormat','Simulation Team');
save_system(model,modelPath);
end

function integrator(model,name,initial,position)
p=[model '/' name]; add_block('built-in/Subsystem',p,'Position',position);
add_block('simulink/Ports & Subsystems/In1',[p '/Physical_Increment'],'Position',[30,100,60,120]);
add_block('simulink/Math Operations/Sum',[p '/Conservation_Update'],'Inputs','++','Position',[145,95,175,125]);
add_block('simulink/Discrete/Unit Delay',[p '/Physical_State'], ...
    'InitialCondition',initial,'SampleTime','1','Position',[245,90,310,130]);
add_block('simulink/Ports & Subsystems/Out1',[p '/Feedback_Previous_State'],'Port','1','Position',[380,100,410,120]);
add_block('simulink/Ports & Subsystems/Out1',[p '/Realized_End_State'],'Port','2','Position',[380,200,410,220]);
add_line(p,'Physical_Increment/1','Conservation_Update/1','autorouting','on');
add_line(p,'Physical_State/1','Conservation_Update/2','autorouting','on');
add_line(p,'Conservation_Update/1','Physical_State/1','autorouting','on');
add_line(p,'Physical_State/1','Feedback_Previous_State/1','autorouting','on');
add_line(p,'Conservation_Update/1','Realized_End_State/1','autorouting','on');
end

function memory(model,name,initial,position)
p=[model '/' name]; add_block('built-in/Subsystem',p,'Position',position);
add_block('simulink/Ports & Subsystems/In1',[p '/Actual_Thermal_Output'],'Position',[30,100,60,120]);
add_block('simulink/Discrete/Unit Delay',[p '/Previous_Output_GW'], ...
    'InitialCondition',initial,'SampleTime','1','Position',[155,90,220,130]);
add_block('simulink/Ports & Subsystems/Out1',[p '/Feedback_Previous_Output'],'Port','1','Position',[380,100,410,120]);
add_block('simulink/Ports & Subsystems/Out1',[p '/Realized_Current_Output'],'Port','2','Position',[380,200,410,220]);
add_line(p,'Actual_Thermal_Output/1','Previous_Output_GW/1','autorouting','on');
add_line(p,'Previous_Output_GW/1','Feedback_Previous_Output/1','autorouting','on');
add_line(p,'Actual_Thermal_Output/1','Realized_Current_Output/1','autorouting','on');
end
