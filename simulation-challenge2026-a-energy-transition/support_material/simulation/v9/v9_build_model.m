% AI-assisted: OpenAI Codex / GPT-6, OpenAI, used 2026-10-18; release date unverified.
function modelPath=v9_build_model(forceRebuild)
% Native scalar conservation network. AI-assisted, OpenAI Codex, 2026-10-18.
if nargin<1,forceRebuild=false;end
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));model='V9_Whitebox_Resource_Network';modelPath=fullfile(root,'simulation','v9',[model '.slx']);
if bdIsLoaded(model),assert(strcmp(get_param(model,'Dirty'),'off') && strcmp(get_param(model,'SimulationStatus'),'stopped'),'Save or close the existing V9 model before building.');close_system(model,0);end
if isfile(modelPath) && ~forceRebuild,load_system(modelPath);return,end
new_system(model);set_param(model,'Solver','FixedStepDiscrete','FixedStep','1','StartTime','0','StopTime','2159','ReturnWorkspaceOutputs','on','SaveOutput','off','Location',[50 50 1500 880]);
add_block('simulink/Sources/From Workspace',[model '/Exogenous_24h_Forecasts'],'VariableName','v9_forecasts','Interpolate','off','OutputAfterFinalValue','Holding final value','Position',[30 70 220 120]);
add_block('simulink/Signal Routing/Mux',[model '/Forecast_and_Previous_State'],'Inputs','2','Position',[300 100 305 300]);
load_system('simulink');ib=find_system('simulink','SearchDepth',2,'Name','Interpreted MATLAB Function');assert(numel(ib)==1);
add_block(ib{1},[model '/Fresh_24h_LP_First_Action'],'MATLABFcn','v9_controller(u)','OutputDimensions','18','Position',[390 125 600 185]);
add_block('built-in/Subsystem',[model '/Physical_Resource_Network'],'Position',[770 105 1050 310]);plant=[model '/Physical_Resource_Network'];
in(plant,'Forecasts',1,[20 40 50 60]);in(plant,'Actions',2,[20 110 50 130]);
select(plant,'Current_Inputs','337','[2:15]',[90 30 200 75]);line(plant,'Forecasts/1','Current_Inputs/1');
add_block('simulink/Signal Routing/Demux',[plant '/Raw_Channels'],'Outputs','14','Position',[250 25 255 350]);line(plant,'Current_Inputs/1','Raw_Channels/1');
add_block('simulink/Signal Routing/Demux',[plant '/Action_Channels'],'Outputs','18','Position',[300 400 305 830]);line(plant,'Actions/1','Action_Channels/1');
names={'Coal','Oil','Gas'};for j=1:3
    p=[plant '/' names{j} '_Stock'];stock(p,j,[430 45+120*(j-1) 660 120+120*(j-1)]);
    line(plant,['Raw_Channels/' num2str(8+j)],[names{j} '_Stock/1']);line(plant,['Raw_Channels/' num2str(11+j)],[names{j} '_Stock/2']);line(plant,['Action_Channels/' num2str(11+j)],[names{j} '_Stock/3']);
    if j~=2,index=1;if j==3,index=2;end;line(plant,['Action_Channels/' num2str(index)],[names{j} '_Stock/4']);end
end
for j=1:2
    name={'Battery_SOC','Pumped_SOC'};p=[plant '/' name{j}];storage(p,j,[430 420+115*(j-1) 660 495+115*(j-1)]);
    line(plant,['Action_Channels/' num2str(6+2*j)],[name{j} '/1']);line(plant,['Action_Channels/' num2str(7+2*j)],[name{j} '/2']);
end
debt([plant '/Shift_Debt'],[430 670 660 745]);line(plant,'Action_Channels/15','Shift_Debt/1');line(plant,'Action_Channels/16','Shift_Debt/2');
ramp([plant '/Thermal_Ramp_Memory'],[770 430 1010 540]);line(plant,'Action_Channels/1','Thermal_Ramp_Memory/1');line(plant,'Action_Channels/2','Thermal_Ramp_Memory/2');
power([plant '/Power_Conservation'],[770 650 1010 750]);line(plant,'Actions/1','Power_Conservation/1');line(plant,'Raw_Channels/1','Power_Conservation/2');
add_block('simulink/Signal Routing/Mux',[plant '/Previous_Resource_State'],'Inputs','8','Position',[1120 70 1125 370]);
add_block('simulink/Signal Routing/Mux',[plant '/Realized_End_State'],'Inputs','8','Position',[1180 420 1185 730]);
sources={'Coal_Stock','Oil_Stock','Gas_Stock','Thermal_Ramp_Memory','Thermal_Ramp_Memory','Battery_SOC','Pumped_SOC','Shift_Debt'};
for j=1:8,previous=1;realized=2;if j==4,previous=1;realized=3;elseif j==5,previous=2;realized=4;end
    line(plant,sprintf('%s/%d',sources{j},previous),sprintf('Previous_Resource_State/%d',j));line(plant,sprintf('%s/%d',sources{j},realized),sprintf('Realized_End_State/%d',j));
end
out(plant,'State_Feedback',1,[1280 200 1310 220]);out(plant,'Realized_States',2,[1280 540 1310 560]);out(plant,'Power_Residual_GW',3,[1280 800 1310 820]);line(plant,'Previous_Resource_State/1','State_Feedback/1');line(plant,'Realized_End_State/1','Realized_States/1');line(plant,'Power_Conservation/1','Power_Residual_GW/1');
set_param(plant,'Location',[40 40 1540 950]);
line(model,'Exogenous_24h_Forecasts/1','Forecast_and_Previous_State/1');line(model,'Physical_Resource_Network/1','Forecast_and_Previous_State/2');line(model,'Forecast_and_Previous_State/1','Fresh_24h_LP_First_Action/1');line(model,'Exogenous_24h_Forecasts/1','Physical_Resource_Network/1');line(model,'Fresh_24h_LP_First_Action/1','Physical_Resource_Network/2');
select(model,'Current_Hour','337','1',[710 390 810 430]);line(model,'Exogenous_24h_Forecasts/1','Current_Hour/1');
add_block('simulink/Signal Routing/Mux',[model '/Actions_and_Realized_States'],'Inputs','2','Position',[1120 100 1125 300]);line(model,'Fresh_24h_LP_First_Action/1','Actions_and_Realized_States/1');line(model,'Physical_Resource_Network/2','Actions_and_Realized_States/2');
add_block('simulink/Signal Routing/Mux',[model '/Read_Only_Observation'],'Inputs','3','Position',[1210 100 1215 440]);line(model,'Current_Hour/1','Read_Only_Observation/1');line(model,'Actions_and_Realized_States/1','Read_Only_Observation/2');line(model,'Physical_Resource_Network/3','Read_Only_Observation/3');
add_block(ib{1},[model '/Trace_Actual_Native_Ports'],'MATLABFcn','v9_trace(u)','OutputDimensions','1','Position',[1280 230 1480 280]);line(model,'Read_Only_Observation/1','Trace_Actual_Native_Ports/1');add_block('simulink/Sinks/Terminator',[model '/Read_Only_Sink'],'Position',[1525 245 1545 265]);line(model,'Trace_Actual_Native_Ports/1','Read_Only_Sink/1');
add_block('simulink/Sinks/To Workspace',[model '/Native_Port_Log'],'VariableName','v9_native_trace','SaveFormat','Array','MaxDataPoints','inf','Position',[1270 390 1470 430]);line(model,'Actions_and_Realized_States/1','Native_Port_Log/1');
note=Simulink.Annotation(model,sprintf('WHITE-BOX MULTI-ENERGY STATE FEEDBACK | 1 native step = 1 hour\nMPC returns 18 actions only. Native Sum/Gain/Unit Delay blocks compute each resource state.\nOpen Physical Resource Network: fuel arrivals vs direct services vs generation; efficiency losses; debt; ramp memory.'));note.Position=[35 520 1490 620];
allBlocks=find_system(model,'Type','Block');for k=1:numel(allBlocks),set_param(allBlocks{k},'FontName','Arial','FontSize','12');if strcmp(get_param(allBlocks{k},'BlockType'),'SubSystem'),set_param(allBlocks{k},'ContentPreviewEnabled','off');end;end
for component={'Coal_Stock','Oil_Stock','Gas_Stock','Battery_SOC','Pumped_SOC','Shift_Debt','Thermal_Ramp_Memory'}
    cp=[plant '/' component{1}];children=find_system(cp,'Type','Block');for k=1:numel(children),if ~strcmp(children{k},cp),set_param(children{k},'FontSize','22');if any(strcmp(get_param(children{k},'BlockType'),{'Gain','UnitDelay'})),set_param(children{k},'NamePlacement','alternate');end;if strcmp(get_param(children{k},'BlockType'),'Sum'),set_param(children{k},'ShowName','off');end;end,end
    noteText='';location=[20 360 800 430];
    switch component{1}
        case 'Coal_Stock',noteText='Power burn = G / (8141 eta_coal), Mtce per 1 h step';
        case 'Oil_Stock',noteText='No oil-fired generation in the declared technology domain';
        case 'Gas_Stock',noteText='Power burn = G / (8141 eta_gas), Mtce per 1 h step';
        case {'Battery_SOC','Pumped_SOC'},noteText=sprintf('Delta SOC = eta_c * charge - discharge / eta_d\nGW x 1 h gives GWh');location=[20 275 800 355];
        case 'Shift_Debt',noteText=sprintf('Debt += shift_out - repay (GWh per 1 h)\n24 h service deadlines are imposed inside the MPC');location=[20 275 800 355];
        case 'Thermal_Ramp_Memory',noteText=sprintf('Previous actual GW feeds the next MPC\nRamp limits are LP inequalities, not plant clipping');location=[20 420 800 500];
    end
    annotation=Simulink.Annotation(cp,noteText);annotation.FontSize=20;annotation.Position=location;
end
save_system(model,modelPath);
end

function stock(p,j,position)
add_block('built-in/Subsystem',p,'Position',position);in(p,'Arrival_Mtce',1,[20 35 50 55]);in(p,'Direct_Demand_Mtce',2,[20 105 50 125]);in(p,'Direct_Unserved_Mtce',3,[20 175 50 195]);
if j~=2,in(p,'Generation_GW',4,[20 265 50 285]);gain='v9_coal_conversion';if j==3,gain='v9_gas_conversion';end;add_block('simulink/Math Operations/Gain',[p '/Electricity_to_Fuel'],'Gain',gain,'Position',[110 255 230 295]);line(p,'Generation_GW/1','Electricity_to_Fuel/1');else,add_block('simulink/Sources/Constant',[p '/No_Oil_Power_Burn'],'Value','0','Position',[110 255 230 295]);end
add_block('simulink/Math Operations/Sum',[p '/Stock_Conservation'],'Inputs','+-+-+','Position',[320 90 350 250]);
set_param([p '/Stock_Conservation'],'Inputs','+-+-+');
add_block('simulink/Discrete/Unit Delay',[p '/Previous_Stock_Mtce'],'InitialCondition',sprintf('v9_initial_fuel(%d)',j),'SampleTime','1','Position',[440 90 560 130]);
line(p,'Arrival_Mtce/1','Stock_Conservation/1');line(p,'Direct_Demand_Mtce/1','Stock_Conservation/2');line(p,'Direct_Unserved_Mtce/1','Stock_Conservation/3');burn='Electricity_to_Fuel';if j==2,burn='No_Oil_Power_Burn';end;line(p,[burn '/1'],'Stock_Conservation/4');line(p,'Previous_Stock_Mtce/1','Stock_Conservation/5');line(p,'Stock_Conservation/1','Previous_Stock_Mtce/1');
out(p,'Prior_Stock',1,[650 105 680 125]);out(p,'End_Stock',2,[650 230 680 250]);line(p,'Previous_Stock_Mtce/1','Prior_Stock/1');line(p,'Stock_Conservation/1','End_Stock/1');set_param(p,'Location',[80 80 1180 740]);
end
function storage(p,j,position)
add_block('built-in/Subsystem',p,'Position',position);in(p,'Charge_GW',1,[20 45 50 65]);in(p,'Discharge_GW',2,[20 155 50 175]);
add_block('simulink/Math Operations/Gain',[p '/Charge_Efficiency'],'Gain',sprintf('v9_eta_charge(%d)',j),'Position',[120 35 240 75]);add_block('simulink/Math Operations/Gain',[p '/Discharge_Loss'],'Gain',sprintf('1/v9_eta_discharge(%d)',j),'Position',[120 145 240 185]);
add_block('simulink/Math Operations/Sum',[p '/SOC_Conservation'],'Inputs','+-+','Position',[320 80 350 150]);add_block('simulink/Discrete/Unit Delay',[p '/Previous_SOC_GWh'],'InitialCondition',sprintf('v9_initial_soc(%d)',j),'SampleTime','1','Position',[440 70 560 110]);
line(p,'Charge_GW/1','Charge_Efficiency/1');line(p,'Discharge_GW/1','Discharge_Loss/1');line(p,'Charge_Efficiency/1','SOC_Conservation/1');line(p,'Discharge_Loss/1','SOC_Conservation/2');line(p,'Previous_SOC_GWh/1','SOC_Conservation/3');line(p,'SOC_Conservation/1','Previous_SOC_GWh/1');out(p,'Prior_SOC',1,[640 75 670 95]);out(p,'End_SOC',2,[640 190 670 210]);line(p,'Previous_SOC_GWh/1','Prior_SOC/1');line(p,'SOC_Conservation/1','End_SOC/1');set_param(p,'Location',[80 80 1180 660]);
end
function debt(p,position)
add_block('built-in/Subsystem',p,'Position',position);in(p,'Shift_Out_GW',1,[25 50 55 70]);in(p,'Repay_GW',2,[25 160 55 180]);add_block('simulink/Math Operations/Sum',[p '/Debt_Conservation'],'Inputs','+-+','Position',[240 80 270 150]);add_block('simulink/Discrete/Unit Delay',[p '/Previous_Debt_GWh'],'InitialCondition','0','SampleTime','1','Position',[380 75 500 115]);line(p,'Shift_Out_GW/1','Debt_Conservation/1');line(p,'Repay_GW/1','Debt_Conservation/2');line(p,'Previous_Debt_GWh/1','Debt_Conservation/3');line(p,'Debt_Conservation/1','Previous_Debt_GWh/1');out(p,'Prior_Debt',1,[590 80 620 100]);out(p,'End_Debt',2,[590 190 620 210]);line(p,'Previous_Debt_GWh/1','Prior_Debt/1');line(p,'Debt_Conservation/1','End_Debt/1');
end
function ramp(p,position)
add_block('built-in/Subsystem',p,'Position',position);for j=1:2,name={'Coal','Gas'};in(p,[name{j} '_Dispatch_GW'],j,[20 50+100*(j-1) 50 70+100*(j-1)]);add_block('simulink/Discrete/Unit Delay',[p '/' name{j} '_Previous_GW'],'InitialCondition',sprintf('v9_initial_thermal(%d)',j),'SampleTime','1','Position',[180 40+100*(j-1) 310 80+100*(j-1)]);line(p,[name{j} '_Dispatch_GW/1'],[name{j} '_Previous_GW/1']);out(p,[name{j} '_Prior_Output'],j,[440 45+100*(j-1) 470 65+100*(j-1)]);out(p,[name{j} '_Current_Output'],j+2,[440 270+100*(j-1) 470 290+100*(j-1)]);line(p,[name{j} '_Previous_GW/1'],[name{j} '_Prior_Output/1']);line(p,[name{j} '_Dispatch_GW/1'],[name{j} '_Current_Output/1']);end
end
function power(p,position)
add_block('built-in/Subsystem',p,'Position',position);in(p,'Actions',1,[20 50 50 70]);in(p,'Demand_GW',2,[20 160 50 180]);add_block('simulink/Math Operations/Gain',[p '/Signed_Physical_Power'],'Gain','[ones(1,7),-1,1,-1,1,zeros(1,3),1,-1,1,-1]','Multiplication','Matrix(K*u)','Position',[120 40 300 80]);add_block('simulink/Math Operations/Sum',[p '/Balance_Residual'],'Inputs','+-','Position',[390 65 420 135]);line(p,'Actions/1','Signed_Physical_Power/1');line(p,'Signed_Physical_Power/1','Balance_Residual/1');line(p,'Demand_GW/1','Balance_Residual/2');out(p,'Residual_GW',1,[520 90 550 110]);line(p,'Balance_Residual/1','Residual_GW/1');
end
function in(p,n,k,pos),add_block('simulink/Ports & Subsystems/In1',[p '/' n],'Port',num2str(k),'Position',pos);end
function out(p,n,k,pos),add_block('simulink/Ports & Subsystems/Out1',[p '/' n],'Port',num2str(k),'Position',pos);end
function select(p,n,w,i,pos),add_block('simulink/Signal Routing/Selector',[p '/' n],'InputPortWidth',w,'Indices',i,'Position',pos);end
function line(p,a,b),add_line(p,a,b,'autorouting','on');end
