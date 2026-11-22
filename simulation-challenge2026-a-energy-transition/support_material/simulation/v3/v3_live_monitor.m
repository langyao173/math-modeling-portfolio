function v3_live_monitor(mode,varargin)
% Read-only UI observer called with newly computed native states.
% Register/reset are controlled by the App; no callback is active in batches.
persistent observer
if strcmp(mode,'register')
    observer=varargin{1};
elseif strcmp(mode,'reset')
    observer=[];
elseif ~isempty(observer)
    observer(mode,varargin{:});
end
end
