function value=v6_session_observer(action,varargin)
% AI-assisted: OpenAI Codex/GPT-6, 2026-10-18. Read-only native step evidence.
% The controller reports the current action and realized end state. This
% observer never writes a physical state or supplies a control action.
persistent evidence
if strcmp(action,'reset')
    evidence=struct('hours',zeros(0,1),'values',zeros(0,26));
elseif strcmp(action,'native')
    mode=varargin{1};
    if strcmp(mode,'hourly')
        hour=varargin{2};state=varargin{3};obs=varargin{4};
        row=[obs(2:19);state(1:3);obs(26:27);state(4:6)]';
        assert(all(isfinite(row)) && numel(row)==26,'V6:Observer','Invalid native observation.');
        if isempty(evidence.hours) || hour>evidence.hours(end)
            evidence.hours(end+1,1)=hour;evidence.values(end+1,:)=row;
        else
            assert(hour==evidence.hours(end),'V6:Observer','Native observation clock reversed.');
            evidence.values(end,:)=row;
        end
    end
elseif ~strcmp(action,'get')
    error('V6:Observer','Unknown observer action.');
end
value=evidence;
end
