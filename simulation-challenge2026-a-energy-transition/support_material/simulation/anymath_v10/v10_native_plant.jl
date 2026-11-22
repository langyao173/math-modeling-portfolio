# AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
# Each call executes one actual native discrete resource transition. Its output,
# rather than the LP's predicted state, becomes the next optimization's input.
const V10_NATIVE_MODEL = "V10_Closed_Loop_Resources"
const V10_NATIVE_RUNS = Ref(0)
const V10_NATIVE_MAX_POWER_RESIDUAL = Ref(0.0)

function v10_native_input!(values)
    global v10_native_u = WorkspaceArray("v10_native_u",
        DataFrame(time=[0.0,1.0], value=[Float64.(values),Float64.(values)]))
end

function v10_build_native()
    if V10_NATIVE_MODEL in getfield.(engee.get_all_models(),:name)
        return V10_NATIVE_MODEL
    end
    engee.create(V10_NATIVE_MODEL)
    engee.set_param!(V10_NATIVE_MODEL,"FixedStep"=>1.0,"StopTime"=>0.0)
    m=V10_NATIVE_MODEL
    function block(lib,name,x,y;w=115,h=38)
        engee.add_block(lib,m*"/"*name;left=x,top=y,width=w,height=h,is_name_visible=true)
    end
    wire(a,b)=engee.add_line(m,a,b)
    v10_native_input!(zeros(25))
    block("/Basic/Sources/From Workspace","Actual_MPC_Action_and_External_Input",30,100;w=230)
    engee.set_param!(m*"/Actual_MPC_Action_and_External_Input","VariableName"=>"v10_native_u",
        "SampleTime"=>"1","Interpolate"=>"false","OutputAfterFinalValue"=>"Holding final value")
    block("/Basic/Signal Routing/Demux","Actions_and_Arrivals",310,70;w=12,h=710)
    engee.set_param!(m*"/Actions_and_Arrivals","Outputs"=>"25")
    wire("Actual_MPC_Action_and_External_Input/1","Actions_and_Arrivals/1")
    block("/Basic/Discrete/Unit Delay","Eight_Resource_States",1250,40;w=210)
    engee.set_param!(m*"/Eight_Resource_States","InitialCondition"=>repr(vcat(V10_INITIAL_FUEL,V10_INITIAL_SOC,zeros(3))),"SampleTime"=>"1")
    block("/Basic/Signal Routing/Demux","Prior_State",490,30;w=12,h=610)
    engee.set_param!(m*"/Prior_State","Outputs"=>"8")
    wire("Eight_Resource_States/1","Prior_State/1")
    block("/Basic/Signal Routing/Mux","Realized_State",1130,80;w=12,h=570)
    engee.set_param!(m*"/Realized_State","Inputs"=>"8")
    wire("Realized_State/1","Eight_Resource_States/1")
    for (j,fuel) in enumerate(V10_FUELS)
        y=110+(j-1)*160;p=uppercasefirst(fuel)
        block("/Basic/Math Operations/Add",p*"_Stock_Balance",920,y;w=50,h=100)
        engee.set_param!(m*"/"*p*"_Stock_Balance","Inputs"=>"++-+-")
        for (port,source) in enumerate(["Prior_State/"*string(j),"Actions_and_Arrivals/"*string(18+j),
                "Actions_and_Arrivals/"*string(21+j),"Actions_and_Arrivals/"*string(11+j)])
            wire(source,p*"_Stock_Balance/"*string(port))
        end
        if j!=2
            block("/Basic/Math Operations/Gain",p*"_Power_to_Fuel",650,y+70;w=155)
            engee.set_param!(m*"/"*p*"_Power_to_Fuel","Gain"=>string(1/(8141*V10_EFFICIENCY[j==1 ? 1 : 2])))
            wire("Actions_and_Arrivals/"*string(j==1 ? 1 : 2),p*"_Power_to_Fuel/1")
        else
            block("/Basic/Sources/Constant",p*"_Power_to_Fuel",650,y+70;w=155)
            engee.set_param!(m*"/"*p*"_Power_to_Fuel","Value"=>"0")
        end
        wire(p*"_Power_to_Fuel/1",p*"_Stock_Balance/5")
        wire(p*"_Stock_Balance/1","Realized_State/"*string(j))
        engee.set_log(m,p*"_Stock_Balance/1")
    end
    for (j,store) in enumerate(V10_STORES)
        y=660+(j-1)*180;p=uppercasefirst(store)
        block("/Basic/Math Operations/Gain",p*"_Charge_Efficiency",650,y;w=155)
        block("/Basic/Math Operations/Gain",p*"_Discharge_Loss",650,y+65;w=155)
        engee.set_param!(m*"/"*p*"_Charge_Efficiency","Gain"=>string(V10_ETA[j]))
        engee.set_param!(m*"/"*p*"_Discharge_Loss","Gain"=>string(1/V10_ETA[j]))
        wire("Actions_and_Arrivals/"*string(6+2*j),p*"_Charge_Efficiency/1")
        wire("Actions_and_Arrivals/"*string(7+2*j),p*"_Discharge_Loss/1")
        block("/Basic/Math Operations/Add",p*"_SOC_Balance",920,y;w=50,h=110)
        engee.set_param!(m*"/"*p*"_SOC_Balance","Inputs"=>"++-")
        wire("Prior_State/"*string(3+j),p*"_SOC_Balance/1")
        wire(p*"_Charge_Efficiency/1",p*"_SOC_Balance/2")
        wire(p*"_Discharge_Loss/1",p*"_SOC_Balance/3")
        wire(p*"_SOC_Balance/1","Realized_State/"*string(3+j))
        engee.set_log(m,p*"_SOC_Balance/1")
    end
    wire("Actions_and_Arrivals/1","Realized_State/6")
    wire("Actions_and_Arrivals/2","Realized_State/7")
    block("/Basic/Math Operations/Add","Service_Debt_Balance",920,1050;w=50,h=110)
    engee.set_param!(m*"/Service_Debt_Balance","Inputs"=>"++-")
    wire("Prior_State/8","Service_Debt_Balance/1")
    wire("Actions_and_Arrivals/15","Service_Debt_Balance/2")
    wire("Actions_and_Arrivals/16","Service_Debt_Balance/3")
    wire("Service_Debt_Balance/1","Realized_State/8")
    block("/Basic/Math Operations/Add","Power_Balance_Residual",1470,220;w=55,h=280)
    engee.set_param!(m*"/Power_Balance_Residual","Inputs"=>repeat("+",7)*"-+-++-+--")
    for (port,input) in enumerate(vcat(collect(1:11),[15,16,17,18,25]))
        wire("Actions_and_Arrivals/"*string(input),"Power_Balance_Residual/"*string(port))
    end
    engee.set_log(m,"Realized_State/1")
    engee.set_log(m,"Power_Balance_Residual/1")
    engee.save(m,joinpath(@__DIR__,m*".engee");force=true)
    V10_NATIVE_RUNS[]=0
    return m
end

function v10_native_step(prior,a,e)
    state=vcat(prior.fuel,prior.soc,prior.thermal,prior.debt)
    u=vcat(a.gen,a.ch[1],a.dis[1],a.ch[2],a.dis[2],a.direct_short,a.out,a.repay,a.unserved,a.spill,e.arrivals,e.direct,e.demand)
    @assert length(u)==25 && length(state)==8
    v10_native_input!(u)
    engee.set_param!(V10_NATIVE_MODEL*"/Eight_Resource_States","InitialCondition"=>repr(state))
    native=engee.run(V10_NATIVE_MODEL)
    q=collect(native["Realized_State.1"])
    @assert nrow(q)==1 && q.time[1]==0.0 "One native evaluation must return exactly one hourly resource transition"
    new=Float64.(q.value[1]);V10_NATIVE_RUNS[]+=1
    residual=maximum(abs.(collect(native["Power_Balance_Residual.1"]).value))
    V10_NATIVE_MAX_POWER_RESIDUAL[]=max(V10_NATIVE_MAX_POWER_RESIDUAL[],residual)
    @assert residual<1e-6 "Native power balance failed at hour $(e.hour)"
    return (fuel=new[1:3],soc=new[4:5],thermal=new[6:7],debt=new[8])
end
