% AI-assisted: OpenAI Codex / GPT-6, OpenAI, used 2026-10-18; release date unverified.
function y=v9_trace(u)
% Read-only sampling downstream of actual native conservation blocks.
persistent evidence
if ischar(u)
    if strcmp(u,'reset'),evidence=struct('hours',zeros(0,1),'values',zeros(0,26),'power_residual_gw',zeros(0,1));end
    y=evidence;return;
end
hour=round(u(1));row=u(2:27)';assert(all(isfinite(row)) && numel(row)==26);
if isempty(evidence.hours) || hour>evidence.hours(end),k=numel(evidence.hours)+1;else,assert(hour==evidence.hours(end));k=numel(evidence.hours);end
evidence.hours(k,1)=hour;evidence.values(k,:)=row;evidence.power_residual_gw(k,1)=u(28);y=0;
end
