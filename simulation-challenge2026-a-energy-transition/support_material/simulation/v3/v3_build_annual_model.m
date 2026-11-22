function modelPath=v3_build_annual_model(rebuild)
% Native closed-loop simulation, not replay of externally calculated controls.
if nargin<1,rebuild=false;end
folder=fileparts(mfilename('fullpath')); addpath(folder);
name='V3_Energy_Transition_ABM'; modelPath=fullfile(folder,[name '.slx']);
if bdIsLoaded(name)
    assert(strcmp(get_param(name,'Dirty'),'off'),'V3:UnsavedModel','Save or close unsaved model edits before rebuilding');
    if ~rebuild,return,end
    close_system(name,0);
end
if isfile(modelPath) && ~rebuild, load_system(modelPath); return, end
new_system(name); set_param(name,'SolverType','Fixed-step','Solver','FixedStepDiscrete','FixedStep','1','StartTime','0','StopTime','35','ReturnWorkspaceOutputs','on','SaveTime','on','TimeSaveName','tout');
set_param(name,'Description','Autonomous annual demand, generation/fuel investment, construction queues, reserves and carbon-policy feedback; fixed calibration only, no annual replay inputs.');
add_block('simulink/Sources/Digital Clock',[name '/Annual clock'],'SampleTime','1','Position',[30 220 100 250]);
stateLedger([name '/Physical and agent states']);
set_param([name '/Physical and agent states'],'Position',[55 65 170 125]);
add_block('simulink/Signal Routing/Mux',[name '/Prior state and clock'],'Inputs','2','Position',[220 85 225 230]);
load_system('simulink'); interpreted=find_system('simulink','SearchDepth',2,'BlockType','MATLABFcn');
assert(numel(interpreted)==1,'V3:Library','Expected one native interpreted MATLAB function block');
add_block(interpreted{1},[name '/Autonomous agent decisions'],'MATLABFcn','v3_annual_step(u)','OutputDimensions','349','Position',[285 95 500 170]);
add_block('simulink/Signal Routing/Demux',[name '/New state and observables'],'Outputs','[217 132]','Position',[545 85 550 195]);
add_block('simulink/Sinks/To Workspace',[name '/State and action trace'],'VariableName','annual_native_trace','SaveFormat','Structure With Time','Position',[620 210 775 245]);
add_block('simulink/Sinks/Scope',[name '/Native annual state dynamics'],'NumInputPorts','3','Position',[795 70 950 180]);
for j=1:3
    selector=[name '/Observable group ' num2str(j)];
    indexSets={'[1 2 3]','[5 7]','[9 10]'};
    add_block('simulink/Signal Routing/Selector',selector,'InputPortWidth','132','Indices',indexSets{j},'Position',[625 45+j*45 695 70+j*45]);
    add_line(name,'New state and observables/2',['Observable group ' num2str(j) '/1'],'autorouting','on');
    add_line(name,['Observable group ' num2str(j) '/1'],['Native annual state dynamics/' num2str(j)],'autorouting','on');
end

function stateLedger(block)
% Explicit state owners. Every output is its preceding-year value.
add_block('built-in/Subsystem',block);
widths=[1 4 9 3 3 repmat(15,1,12) 4 7 1 1 3 1];
names={'Useful service','User equipment shares','Power and storage capacities', ...
    'Domestic fuel capacities','Coal oil gas reserves', ...
    'Coal project pipeline','Gas project pipeline','Nuclear project pipeline', ...
    'Hydro project pipeline','Wind project pipeline','Solar project pipeline', ...
    'Biomass project pipeline','Battery project pipeline','Pumped project pipeline', ...
    'Coal supplier pipeline','Oil supplier pipeline','Gas supplier pipeline', ...
    'Service shortage memory','Generator utilisation memory','Curtailment memory', ...
    'Government carbon price','Import channels','Electricity market price'};
assert(sum(widths)==217 && numel(names)==numel(widths));
add_block('simulink/Ports & Subsystems/In1',[block '/Newly computed state'],'Position',[30 400 60 420]);
add_block('simulink/Signal Routing/Demux',[block '/Separate state owners'], ...
    'Outputs',mat2str(widths),'Position',[120 70 125 900]);
add_block('simulink/Signal Routing/Mux',[block '/Previous year state'], ...
    'Inputs',num2str(numel(widths)),'Position',[1190 70 1195 900]);
add_block('simulink/Ports & Subsystems/Out1',[block '/State feedback'],'Position',[1270 400 1300 420]);
add_line(block,'Newly computed state/1','Separate state owners/1');
add_line(block,'Previous year state/1','State feedback/1');
first=1;
for j=1:numel(widths)
    column=floor((j-1)/12);row=mod(j-1,12);x=260+column*470;y=35+row*75;
    initial=sprintf('V3_ANNUAL_CONTEXT.initial_state(%d:%d)',first,first+widths(j)-1);
    add_block('simulink/Discrete/Unit Delay',[block '/' names{j}], ...
        'InitialCondition',initial,'SampleTime','1','Position',[x y x+245 y+35]);
    add_line(block,['Separate state owners/' num2str(j)],[names{j} '/1'],'autorouting','on');
    add_line(block,[names{j} '/1'],['Previous year state/' num2str(j)],'autorouting','on');
    first=first+widths(j);
end
note=Simulink.Annotation(block,'Each project pipeline stores 15 future commissioning slots. Annual decisions shift the slots and insert new orders; these blocks retain the real previous-year queues.');
note.Position=[170 970];
end
configuration=get_param([name '/Native annual state dynamics'],'ScopeConfiguration');configuration.LayoutDimensions=[3 1];
add_line(name,'Physical and agent states/1','Prior state and clock/1','autorouting','on');
add_line(name,'Annual clock/1','Prior state and clock/2','autorouting','on');
add_line(name,'Prior state and clock/1','Autonomous agent decisions/1','autorouting','on');
add_line(name,'Autonomous agent decisions/1','New state and observables/1','autorouting','on');
add_line(name,'New state and observables/1','Physical and agent states/1','autorouting','on');
add_line(name,'Autonomous agent decisions/1','State and action trace/1','autorouting','on');
annotation=Simulink.Annotation(name,'217 endogenous states: useful-service demand and equipment shares; nine power assets; three fuel suppliers; fuel reserves; twelve 15-stage construction queues; shortage/utilisation/curtailment signals; carbon policy and market price, import channels.');
annotation.Position=[25 290];
set_param(name,'Location',[100 100 1100 600]); save_system(name,modelPath);
end
