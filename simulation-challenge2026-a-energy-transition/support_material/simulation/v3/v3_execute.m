function output=v3_execute(input,stage,pace)
% Render only between genuine paused Simulink steps, outside the solver.
% MathWorks Simulation.step API, available since R2024a.
if pace<=0,output=sim(input);return,end
session=simulation(input);
session=setModelParameter(session,EnablePacing="off");
cleanup=onCleanup(@()stopIfActive(session));
if strcmp(stage,'annual'),chunk=6;else,chunk=72;end
last=false;
while ~last
    last=step(session,NumberOfSteps=chunk);
    v3_live_monitor('paint',stage);
    drawnow;
    pause(.2);
end
if session.Status=="inactive",output=session.SimulationOutput;else,output=stop(session);end
end

function stopIfActive(session)
if session.Status~="inactive",stop(session);end
end
