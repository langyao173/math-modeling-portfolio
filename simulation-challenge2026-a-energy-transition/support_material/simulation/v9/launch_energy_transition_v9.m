% AI-assisted: OpenAI Codex / GPT-6, OpenAI, used 2026-10-18; release date unverified.
function app=launch_energy_transition_v9()
% Native execution and already-executed experiment replay are separate modes.
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
addpath(folder);addpath(fullfile(root,'simulation','v6'));addpath(fullfile(root,'simulation','v3'));addpath(fullfile(root,'simulation','v4'));
prior=v9_session('get');
assert(strcmp(prior.status,'not_started'),'V9:ExistingSession','已有原生会话，请先在原窗口关闭它。V9 不会接管或覆盖。');
protocol=jsondecode(fileread(fullfile(root,'config','v6_resilience.json')));
caseInherited=v6_resilience_case([],"inherited",2160,"repeated");
replay.inherited=readtable(fullfile(root,'outputs','v6','native','resilience_inherited.csv'));
replay.contract=readtable(fullfile(root,'outputs','v6','native','resilience_contract.csv'));
assert(height(replay.inherited)==2160 && height(replay.contract)==2160,'V9:Replay','回放需要两条完整的已执行 2160 小时轨迹。');
assert(isequal(replay.inherited.hour,(0:2159)') && isequal(replay.contract.hour,(0:2159)'),'V9:ReplayClock','回放轨迹时钟异常。');
ink=[.149 .204 .239];blue=[.141 .455 .667];teal=[.086 .529 .482];orange=[.808 .490 .208];purple=[.502 .388 .624];gray=[.471 .522 .561];
fuelColors=[gray;orange;purple];techColors=[gray;purple;.43 .62 .70;.27 .56 .76;teal;.83 .68 .31;.48 .64 .41;blue];
fig=uifigure('Name','多能源转型过程实验室 V9','Position',[20 35 1600 900],'Color',[1 1 1],'CloseRequestFcn',@closeApp);
fig.Tag='V9_Energy_Process_Lab';
outer=uigridlayout(fig,[3 2]);outer.RowHeight={78,'1x',26};outer.ColumnWidth={246,'1x'};outer.Padding=[16 10 16 8];outer.RowSpacing=8;outer.ColumnSpacing=16;outer.BackgroundColor=[1 1 1];
head=uigridlayout(outer,[2 3]);head.Layout.Column=[1 2];head.RowHeight={35,26};head.ColumnWidth={'1x',195,240};head.Padding=[0 0 0 0];head.BackgroundColor=[1 1 1];
titleLabel=uilabel(head,'Text','多能源协同 · 白盒状态与约束实验室','FontSize',23,'FontWeight','bold','FontColor',ink);titleLabel.Layout.Row=1;titleLabel.Layout.Column=1;
mode=uidropdown(head,'Items',{'原生白盒求解','V6 已执行轨迹回放'},'ItemsData',{'live','replay'},'Value','live','ValueChangedFcn',@modeChanged,'FontSize',15);mode.Layout.Row=1;mode.Layout.Column=2;
annualButton=uibutton(head,'Text','打开原生资源守恒网络','ButtonPushedFcn',@openAnnual,'FontSize',13);annualButton.Layout.Row=1;annualButton.Layout.Column=3;
clockLabel=uilabel(head,'Text','原生求解 | 尚未初始化','FontSize',17,'FontWeight','bold','FontColor',blue);clockLabel.Layout.Row=2;clockLabel.Layout.Column=[1 2];
eventLabel=uilabel(head,'Text','初始运行 → 冲击 1 → 恢复 → 冲击 2','FontSize',12,'FontColor',gray);eventLabel.Layout.Row=2;eventLabel.Layout.Column=3;
left=uigridlayout(outer,[22 1]);left.RowHeight={23,30,23,30,23,30,34,34,34,34,24,30,25,32,23,30,30,26,34,30,50,'1x'};left.RowSpacing=5;left.Padding=[0 0 0 0];left.BackgroundColor=[1 1 1];
labelLeft('下次初始化 · 供应方式',1);
supply=uidropdown(left,'Items',{'继承供应','付费补库合同'},'ItemsData',{'inherited','contract'},'Value','inherited','FontSize',13);supply.Layout.Row=2;
labelLeft('下次初始化 · 运行策略',3);
strategy=uidropdown(left,'Items',{'协同滚动调度 · 24 h','分散储能规则 · 24 h'},'ItemsData',{'native_coordinated_mpc','native_decentralized_mpc'},'Value','native_coordinated_mpc','FontSize',13);strategy.Layout.Row=4;
labelLeft('下次初始化 · 运行时长 / h',5);
horizon=uispinner(left,'Limits',[24 2160],'Value',2160,'Step',24,'RoundFractionalValues','on');horizon.Layout.Row=6;
prepareButton=button('初始化新原生会话',7,@(~,~)prepareLive());
stepButton=button('求解下一小时',8,@(~,~)stepLive(1));
dayButton=button('求解并推进 24 小时',9,@(~,~)stepLive(24));
playButton=button('连续原生求解',10,@(~,~)play());
pauseButton=button('暂停',11,@(~,~)pause());
labelLeft('回放 · 供应方式',12);
replaySupply=uidropdown(left,'Items',{'继承供应','付费补库合同'},'ItemsData',{'inherited','contract'},'Value','inherited','ValueChangedFcn',@(~,~)render());replaySupply.Layout.Row=13;
pair= uicheckbox(left,'Text','同一小时配对另一供应方式','Value',true,'ValueChangedFcn',@(~,~)render(),'FontSize',12);pair.Layout.Row=14;
labelLeft('回放速度 / 每次刷新推进小时',15);
speed=uidropdown(left,'Items',{'1 h','6 h','24 h','72 h'},'ItemsData',[1 6 24 72],'Value',24);speed.Layout.Row=16;
eventJump=uidropdown(left,'Items',{'初始时刻 · h=0','冲击 1 起点 · h=120','合同到货起点 · h=648','冲击 2 起点 · h=1200','小时峰值窗口 · h=1338','末小时 · h=2159'},'ItemsData',[0 120 648 1200 1338 2159],'Value',0,'ValueChangedFcn',@(~,~)seekReplay(eventJump.Value));eventJump.Layout.Row=17;
zoom=uidropdown(left,'Items',{'已执行时间窗口','完整 90 天设计轴','首冲击 5–6 天','第 55–57 天小时放大'},'ItemsData',{'prefix','full','first','zoom'},'Value','prefix','ValueChangedFcn',@(~,~)render());zoom.Layout.Row=18;
replayPlay=button('播放已执行轨迹',19,@(~,~)play());
timeSlider=uislider(left,'Limits',[0 2159],'Value',0,'MajorTicks',[0 1200 2159],'MinorTicks',[],'ValueChangingFcn',@(~,e)seekReplay(e.Value),'ValueChangedFcn',@(~,e)seekReplay(e.Value));timeSlider.Layout.Row=20;
modeNote=uilabel(left,'Text',{'默认原生求解：只显示已执行前缀。';'年度 2060 快照单向传入小时模型。'},'FontSize',11,'FontColor',gray,'WordWrap','on');modeNote.Layout.Row=21;
status=uilabel(left,'Text','初始化不会推进求解。','WordWrap','on','FontSize',12,'VerticalAlignment','top','FontColor',ink);status.Layout.Row=22;
main=uigridlayout(outer,[3 2]);main.RowHeight={345,'1x','1x'};main.ColumnWidth={'1.35x','1x'};main.Padding=[0 0 0 0];main.RowSpacing=12;main.ColumnSpacing=14;main.BackgroundColor=[1 1 1];
currentGrid=uigridlayout(main,[3 1]);currentGrid.RowHeight={26,'1x',52};currentGrid.Padding=[0 0 0 0];currentGrid.BackgroundColor=[1 1 1];
supplyTitle=uilabel(currentGrid,'Text','当期电力构成 / GW','FontSize',15,'FontWeight','bold','FontColor',ink);
axSupply=uiaxes(currentGrid);balanceLabel=uilabel(currentGrid,'Text','—','FontSize',12,'WordWrap','on','FontColor',ink);
resourceGrid=uigridlayout(main,[4 1]);resourceGrid.RowHeight={24,75,98,'1x'};resourceGrid.Padding=[0 0 0 0];resourceGrid.BackgroundColor=[1 1 1];
resourceTitle=uilabel(resourceGrid,'Text','资源状态 · 物理裕量与约束代价','FontSize',15,'FontWeight','bold','FontColor',ink);
axSOC=uiaxes(resourceGrid);fuelTable=uitable(resourceGrid,'ColumnName',{'燃料','库存 / Mtce','事件前比 / %','配对差 / Mtce'},'RowName',{},'ColumnWidth',{46,100,110,108},'FontName','Microsoft YaHei','FontSize',12,'Data',cell(3,4));
constraintTable=uitable(resourceGrid,'ColumnName',{'约束类别','当前诊断'},'RowName',{},'ColumnWidth',{140,280},'FontName','Microsoft YaHei','FontSize',11,'Data',cell(5,2));
axPower=uiaxes(main);axPower.Layout.Row=2;axPower.Layout.Column=[1 2];
axFuel=uiaxes(main);axFuel.Layout.Row=3;axFuel.Layout.Column=[1 2];
footer=uilabel(outer,'Text','折中投资方案 plan_02349 · 2060 年末快照 | 共同 24 h 信息 | 固定设计冲击：第 5–25 天、第 50–70 天','FontSize',12,'FontColor',gray);footer.Layout.Row=3;footer.Layout.Column=[1 2];
for a=[axSupply axSOC axPower axFuel]
    a.FontName='Microsoft YaHei';a.FontSize=11;a.XColor=gray;a.YColor=gray;a.Box='off';a.Color=[1 1 1];a.GridColor=[.88 .90 .92];a.GridAlpha=.65;a.Toolbar.Visible='off';
    a.TitleHorizontalAlignment='left';
end
live=prior;ownsLive=false;busy=false;playing=false;replayHour=0;closing=false;labToken=char(java.util.UUID.randomUUID());
baseRows=left.RowHeight;
playTimer=timer('ExecutionMode','fixedSpacing','Period',.2,'BusyMode','drop','TimerFcn',@tick,'ErrorFcn',@timerError);
app=struct('figure',fig,'mode',mode,'supply',supply,'strategy',strategy,'horizon',horizon,'replaySupply',replaySupply,'pair',pair,'speed',speed,'eventJump',eventJump,'zoom',zoom,'slider',timeSlider, ...
    'prepareButton',prepareButton,'stepButton',stepButton,'dayButton',dayButton,'playButton',playButton,'pauseButton',pauseButton,'replayPlay',replayPlay,'annualButton',annualButton, ...
    'setMode',@setMode,'prepareLive',@prepareLive,'stepLive',@stepLive,'play',@play,'pause',@pause,'seekReplay',@seekReplay,'advanceReplay',@advanceReplay,'getState',@getState,'close',@closeApp, ...
    'axes',struct('supply',axSupply,'soc',axSOC,'power',axPower,'fuel',axFuel),'clockLabel',clockLabel,'status',status,'fuelTable',fuelTable,'constraintTable',constraintTable,'timer',playTimer);
render();refreshControls();

    function h=labelLeft(value,row)
        h=uilabel(left,'Text',value,'FontSize',11,'FontColor',gray);h.Layout.Row=row;
    end
    function h=button(value,row,callback)
        h=uibutton(left,'Text',value,'ButtonPushedFcn',callback,'FontSize',12);h.Layout.Row=row;
    end
    function modeChanged(varargin),pause();if strcmp(mode.Value,'live'),zoom.Value='prefix';else,zoom.Value='full';end;render();refreshControls();end %#ok<INUSD>
    function setMode(value),mode.Value=value;modeChanged();end
    function prepareLive()
        assert(strcmp(mode.Value,'live') && ~busy && ~playing,'V9:Mode','暂停原生求解后才能初始化。');
        actual=v9_session('get');assert(strcmp(actual.status,'not_started') || isOurs(actual),'V9:Ownership','其他窗口已有小时会话；请先在该窗口关闭，V9 未覆盖它。');
        c=v6_resilience_case([],string(supply.Value),horizon.Value,"repeated");
        c.v9_owner_token=labToken;
        live=v9_session('prepare',c,strategy.Value);ownsLive=true;render();refreshControls();
    end
    function s=stepLive(n)
        assert(strcmp(mode.Value,'live') && ~busy,'V9:Mode','下一小时按钮只适用于原生模式。');
        assert(any(strcmp(live.status,{'prepared','paused'})),'V9:Session','先初始化原生会话。');
        assert(isOurs(v9_session('get')),'V9:Ownership','内核会话已由其他窗口更改；V9 未推进它。');
        busy=true;refreshControls();status.Text='正在执行实际 Simulink 求解。';drawnow;
        cleanup=onCleanup(@finishStep); %#ok<NASGU>
        try
            live=v9_session('step',min(n,live.case.hours-live.executed_steps));s=live;render();
        catch failure
            live=v9_session('get');status.Text=failure.message;pause();rethrow(failure);
        end
    end
    function finishStep(),busy=false;if isvalid(fig),refreshControls();end,end
    function play()
        assert(~busy && ~playing,'V9:Busy','当前求解或播放尚未暂停。');
        if strcmp(mode.Value,'live'),assert(any(strcmp(live.status,{'prepared','paused'})),'V9:Session','先初始化原生会话。');end
        playing=true;start(playTimer);refreshControls();
    end
    function pause()
        playing=false;if strcmp(playTimer.Running,'on'),stop(playTimer);end
        if isvalid(fig),refreshControls();end
    end
    function tick(varargin) %#ok<INUSD>
        if ~playing || closing,return,end
        if strcmp(mode.Value,'replay')
            advanceReplay(speed.Value);
            if replayHour>=2159,pause();end
        else
            stepLive(24);
            if ~any(strcmp(live.status,{'prepared','paused'})),pause();end
        end
    end
    function timerError(~,event)
        pause();if isvalid(fig),status.Text=['执行停止：' event.Data.Message];end
    end
    function seekReplay(hour)
        assert(strcmp(mode.Value,'replay'),'V9:Mode','时间跳转只适用于已执行实验回放。');
        replayHour=max(0,min(2159,round(hour)));timeSlider.Value=replayHour;render();
    end
    function advanceReplay(n),seekReplay(replayHour+n);end
    function state=getState()
        state=struct('mode',mode.Value,'playing',playing,'busy',busy,'replay_hour',replayHour,'live_status',live.status,'live_executed_steps',live.executed_steps,'owns_live',ownsLive);
        if ownsLive,state.live_trace=live.trace;state.live_case=live.case;end
    end
    function refreshControls()
        isLive=strcmp(mode.Value,'live');idle=~busy && ~playing;active=any(strcmp(live.status,{'prepared','paused'}));
        prepareButton.Enable=isLive && idle;stepButton.Enable=isLive && idle && active;dayButton.Enable=stepButton.Enable;playButton.Enable=stepButton.Enable;
        for h=[supply strategy horizon],h.Enable=isLive && idle;end
        for h=[replaySupply speed eventJump timeSlider pair],h.Enable=~isLive && ~busy;end
        replayPlay.Enable=~isLive && idle;pauseButton.Enable=playing || busy;mode.Enable=~busy;annualButton.Enable=idle;zoom.Enable=~busy;
        rows=baseRows;hidden=12:20;if ~isLive,hidden=1:10;end;rows(hidden)={0};rows{18}=30;rows{20}=46;if isLive,rows{20}=0;end;left.RowHeight=rows;
        for child=left.Children'
            row=child.Layout.Row;child.Visible=~any(ismember(row,hidden)) || row==18;
        end
    end
    function render()
        isReplay=strcmp(mode.Value,'replay');T=table();other=table();c=caseInherited;
        if isReplay
            T=replay.(replaySupply.Value);n=replayHour+1;
            if pair.Value,alternate='contract';if strcmp(replaySupply.Value,'contract'),alternate='inherited';end;other=replay.(alternate);end
            prefix='V6 已执行轨迹回放';policy=policyName(replaySupply.Value);strategyName='协同滚动调度';total=2160;
            modeNote.Text={'回放已执行的原生 2160 h 记录。';'完整轨迹可见；不启动求解器。'};
        else
            prefix='原生白盒求解';total=horizon.Value;n=0;policy='尚未冻结';strategyName='尚未初始化';
            if ownsLive,T=live.trace;n=live.executed_steps;c=live.case;total=c.hours;policy=policyName(char(c.supply_policy));strategyName='协同滚动调度';if contains(live.strategy,'decentralized'),strategyName='分散储能规则';end,end
            modeNote.Text={'连续求解每块 24 h，块边界暂停。';'图中只显示实际已求解的前缀。'};
        end
        footer.Text=sprintf('plan_02349 · 折中 / 2060 | 当前冻结时长 %d h | 两次固定冲击：第 5–25、50–70 天 | %s',total,prefix);
        if n==0
            clockLabel.Text=sprintf('%s | 已完成 0 / %d h · 尚未推进',prefix,total);eventLabel.Text='初始运行 · 尚未执行第一小时';
            for a=[axSupply axSOC axPower axFuel],clearAxis(a);end
            supplyTitle.Text='当期电力构成 / GW';balanceLabel.Text='初始化 → 求解 → 展示真实状态';fuelTable.Data=cell(3,4);constraintTable.Data=cell(5,2);
            blankTimeline(axPower,'电力短缺 / GW');blankTimeline(axFuel,'燃料库存 / Mtce');status.Text=sprintf('当前快照：折中 / 2060\n%s · %s\n尚未执行。',policy,strategyName);drawnow;return;
        end
        r=T(n,:);hour=r.hour;day=hour/24;event=eventAt(hour);eventText='初始运行';if event>0,eventText=sprintf('冲击 %d 后的恢复',event);if hour<protocol.events(event,2),eventText=sprintf('冲击 %d 进行中',event);end,end
        clockLabel.Text=sprintf('%s | 已执行 %d / %d h · 当前 h=%d · 第 %.2f 天',prefix,n,total,hour,day);if isReplay,clockLabel.Text=sprintf('%s | h=%d / 2159 · 第 %.2f 天',prefix,hour,day);end;eventLabel.Text=eventText;
        supplyTitle.Text=sprintf('当期电力供给 / GW · %s',policy);
        gen=r{1,{'coal_gw','gas_gw','nuclear_gw','hydro_gw','wind_gw','solar_gw','biomass_gw'}};dis=r.battery_discharge_gw+r.pumped_discharge_gw;charge=r.battery_charge_gw+r.pumped_charge_gw;
        drawTopology(gen,dis,charge,r.served_power_gw,max(r.unserved_gw,0),r.spill_gw,r.demand_gw);
        balance=r.demand_gw-(sum(gen)+dis-charge+r.shift_out_gw-r.shift_in_gw+r.unserved_gw-r.spill_gw);
        balanceLabel.Text=sprintf('实际服务 %.1f  |  未供电 %.1f  |  充电 %.1f  |  弃电 %.1f GW\n移出 %.1f / 偿还 %.1f GW  ·  电力守恒残差 %.1e GW',r.served_power_gw,max(r.unserved_gw,0),charge,r.spill_gw,r.shift_out_gw,r.shift_in_gw,balance);
        soc=[r.battery_soc_gwh,r.pumped_soc_gwh];cap=[c.storage.battery.energy_gwh,c.storage.pumped.energy_gwh];pct=100*soc./cap;
        clearAxis(axSOC);barh(axSOC,1:2,pct,.45,'FaceColor',teal,'EdgeColor','none');axSOC.YTick=1:2;axSOC.YTickLabel={'电池','抽蓄'};axSOC.XTick=[0 50 100];xlim(axSOC,[0 125]);ylim(axSOC,[.5 2.5]);
        for j=1:2,text(axSOC,2,j,sprintf('%.1f%% · %.1f / %.1f GWh',pct(j),soc(j),cap(j)),'FontName','Microsoft YaHei','FontSize',10,'Color',ink);end
        stocks=r{1,{'coal_stock_mtce','oil_stock_mtce','gas_stock_mtce'}};ratios=nan(1,3);ref=nan(1,3);
        if event>0,ref=T{protocol.events(event,1),{'coal_stock_mtce','oil_stock_mtce','gas_stock_mtce'}};ratios=100*stocks./ref;end
        delta=nan(1,3);if ~isempty(other),delta=stocks-other{n,{'coal_stock_mtce','oil_stock_mtce','gas_stock_mtce'}};end
        names={'煤','油','气'};cells=cell(3,4);
        for j=1:3,cells{j,1}=names{j};cells{j,2}=sprintf('%.3f',stocks(j));cells{j,3}='尚无冲击';if event>0,cells{j,3}=sprintf('%.2f',ratios(j));end;cells{j,4}='—';if ~isempty(other),cells{j,4}=sprintf('%+.3f',delta(j));end,end
        fuelTable.Data=cells;
        show=T;if ~isReplay,show=T(1:n,:);end
        clearAxis(axPower);hold(axPower,'on');plot(axPower,show.hour/24,max(show.unserved_gw,0),'Color',blue,'LineWidth',1.35,'DisplayName',policy);
        if ~isempty(other),plot(axPower,other.hour/24,max(other.unserved_gw,0),'--','Color',gray,'LineWidth',1,'DisplayName',policyName(alternate));end
        powerMax=max(1,max(show.unserved_gw)*1.12);if ~isempty(other),powerMax=max(powerMax,max(other.unserved_gw)*1.12);end;ylim(axPower,[0 powerMax]);timeline(axPower,day);ylabel(axPower,'未供电 / GW');title(axPower,'小时运行 · 电力短缺（每点为一小时）','HorizontalAlignment','left','FontSize',13,'FontWeight','bold');legend(axPower,'Location','northeast','Orientation','horizontal','Box','off');hold(axPower,'off');
        clearAxis(axFuel);hold(axFuel,'on');for j=1:3
            field=string(c.fuels(j))+"_stock_mtce";plot(axFuel,show.hour/24,show.(field),'Color',fuelColors(j,:),'LineWidth',1.4,'DisplayName',names{j});
            if ~isempty(other),plot(axFuel,other.hour/24,other.(field),'--','Color',fuelColors(j,:),'LineWidth',.9,'HandleVisibility','off');end
            if event>0,yline(axFuel,.95*ref(j),':','Color',fuelColors(j,:),'HandleVisibility','off','LineWidth',.8);end
        end
        fuelTitle='燃料库存 · 实际已求解前缀';if isReplay,fuelTitle='燃料库存 · 当前供应方式';if ~isempty(other),fuelTitle='燃料库存 · 实线为当前供应方式，虚线为同时间配对';end,end
        ylim(axFuel,[0,max(1,1.08*max([T{:,{'coal_stock_mtce','oil_stock_mtce','gas_stock_mtce'}}],[],'all'))]);timeline(axFuel,day);ylabel(axFuel,'库存 / Mtce');xlabel(axFuel,'设计实验时间 / 天');title(axFuel,fuelTitle,'HorizontalAlignment','left','FontSize',13,'FontWeight','bold');legend(axFuel,'Location','northeast','Orientation','horizontal','Box','off');hold(axFuel,'off');
        referenceText='尚无已发生冲击，不显示恢复基准。';if event>0,referenceText=sprintf('恢复参照：冲击 %d 前一小时库存；点线 = 95%%。',event);end
        status.Text=sprintf('当前实际：折中 / 2060\n%s · %s\n%s\n配对差 = 当前方式 − 另一方式。\n当前未供电 %.2f GW',policy,strategyName,referenceText,max(r.unserved_gw,0));
        constraintTable.Data=cell(5,2);
        if ~isReplay
            d=v9_controller('get');
            if isfield(d,'hour') && d.hour==hour
                genNames={'煤','气','核','水','风','光','生物质'};rampNames={'煤升坡','煤降坡','气升坡','气降坡'};storeNames={'电池','抽蓄'};fuelNames={'煤','油','气'};
                constraintTable.Data={'储能功率裕量',sprintf('电池 %.3f / 抽蓄 %.3f GW',d.storage_power_margin_gw);'发电上界正影子价',activeNames(genNames,d.upper_generation_dual);'爬坡正影子价',activeNames(rampNames,d.ramp_dual);'SOC下界正影子价',activeNames(storeNames,d.soc_lower_dual);'燃料下界正影子价',activeNames(fuelNames,d.fuel_lower_dual)};
                status.Text=sprintf('实际方案：折中 / 2060\n%s · %s\n%s\n服务债务 %.1f GWh\n电力残差 %.1e GW\n右侧分别报告物理裕量与正对偶代价，不将零对偶价当作未贴界。',policy,strategyName,referenceText,r.shift_debt_gwh,balance);
            end
        end
        drawnow;
    end
    function textValue=activeNames(names,values)
        chosen=names(values>1e-7);textValue='—';if ~isempty(chosen),textValue=strjoin(chosen,' / ');end
    end
    function drawTopology(g,dis,ch,served,unserved,spill,demand)
        clearAxis(axSupply);axis(axSupply,[0 1 0 1]);axis(axSupply,'off');hold(axSupply,'on');
        vals=[g(1),g(2),g(5)+g(6),g(3)+g(4)+g(7),dis];labels={'煤电','气电','风光','核水生物质','储能放电'};
        colors=[gray;purple;teal;[.27 .56 .76];blue];ys=[.86 .68 .50 .32 .14];
        for k=1:5
            rectangle(axSupply,'Position',[.01 ys(k)-.055 .28 .11],'FaceColor',[1 1 1],'EdgeColor',colors(k,:),'LineWidth',1);
            text(axSupply,.025,ys(k),sprintf('%s  %.1f GW',labels{k},vals(k)),'FontSize',10,'FontName','Microsoft YaHei','Color',ink);
            width=.4+7*max(vals(k),0)/max(1,demand);col=colors(k,:);if vals(k)<1e-6,col=[.85 .87 .89];end
            quiver(axSupply,.30,ys(k),.20,0,0,'Color',col,'LineWidth',width,'MaxHeadSize',.15);
        end
        if ch>1e-6,quiver(axSupply,.50,.10,-.20,0,0,'Color',blue,'LineWidth',.4+7*ch/max(1,demand),'MaxHeadSize',.15);end
        plot(axSupply,[.51 .51],[.08 .88],'-','Color',ink,'LineWidth',1.2);
        text(axSupply,.53,.91,'电力母线','FontSize',11,'FontName','Microsoft YaHei','Color',ink);
        quiver(axSupply,.52,.70,.22,0,0,'Color',blue,'LineWidth',.4+7*served/max(1,demand),'MaxHeadSize',.12);
        rectangle(axSupply,'Position',[.76 .61 .235 .18],'FaceColor',[1 1 1],'EdgeColor',blue);
        text(axSupply,.78,.70,sprintf('实际服务\n%.1f GW',served),'FontSize',11,'FontName','Microsoft YaHei','Color',ink);
        text(axSupply,.66,.48,sprintf('需求 %.1f GW',demand),'FontSize',11,'FontName','Microsoft YaHei','Color',ink);
        shortageColor=gray;if unserved>1e-5,shortageColor=[.76 .28 .22];end
        text(axSupply,.66,.34,sprintf('未供电 %.1f GW',unserved),'FontSize',11,'FontWeight','bold','FontName','Microsoft YaHei','Color',shortageColor);
        text(axSupply,.66,.20,sprintf('储能充电 %.1f GW\n弃电 %.1f GW',ch,spill),'FontSize',10,'FontName','Microsoft YaHei','Color',ink);
        text(axSupply,.01,.015,'线宽随当期功率变化；燃料库存另计 Mtce','FontSize',9,'FontName','Microsoft YaHei','Color',gray);hold(axSupply,'off');
    end
    function blankTimeline(a,value),xlim(a,[0 90]);ylim(a,[0 1]);ylabel(a,value);xlabel(a,'设计实验时间 / 天');grid(a,'on');end
    function clearAxis(a)
        % cla without reset retains hidden paired traces and event lines.
        % Clearing them is essential when returning from replay to a live prefix.
        cla(a,'reset');a.FontName='Microsoft YaHei';a.FontSize=11;a.XColor=gray;a.YColor=gray;a.Box='off';a.Color=[1 1 1];a.GridColor=[.88 .90 .92];a.GridAlpha=.65;a.Toolbar.Visible='off';a.TitleHorizontalAlignment='left';a.NextPlot='add';
    end
    function timeline(a,day)
        xlim(a,[0 90]);if strcmp(zoom.Value,'prefix'),xlim(a,[0 max(1,day+.25)]);elseif strcmp(zoom.Value,'first'),xlim(a,[5 6]);elseif strcmp(zoom.Value,'zoom'),xlim(a,[55 57]);end;grid(a,'on');
        yl=ylim(a);for j=1:2,p=patch(a,protocol.events(j,[1 2 2 1])/24,yl([1 1 2 2]),[.95 .90 .82],'FaceAlpha',.6,'EdgeColor','none','HandleVisibility','off');uistack(p,'bottom');end
        xline(a,day,'-','Color',ink,'LineWidth',1.1,'HandleVisibility','off');
    end
    function event=eventAt(hour),event=find(hour>=protocol.events(:,1),1,'last');if isempty(event),event=0;end,end
    function name=policyName(key),name='继承供应';if strcmp(key,'contract'),name='付费补库合同';end,end
    function yes=isOurs(s),yes=isfield(s.case,'v9_owner_token') && strcmp(s.case.v9_owner_token,labToken);end
    function openAnnual(~,~)
        assert(~busy,'V9:Annual','当前求解步完成后再打开模型。');pause();
        model='V9_Whitebox_Resource_Network';if ~bdIsLoaded(model),v9_build_model();end;open_system([model '/Physical_Resource_Network']);
    end
    function closeApp(varargin) %#ok<INUSD>
        assert(~busy,'V9:Busy','正在执行求解，请等待当前步结束。');closing=true;pause();delete(playTimer);
        if ownsLive && isOurs(v9_session('get')),v9_session('close');end
        if isvalid(fig),delete(fig);end
    end
end
