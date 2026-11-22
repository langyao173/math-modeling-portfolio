# AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
using TOML
const V10_GUI_SESSION = Ref{Any}(nothing)
const V10_GUI_ID = Ref(0)
const V10_BATCH_RUNNING = Ref(false)
const V10_SAVED_SESSIONS = Dict{String,Any}()
const V10_SAVED_META = Dict{String,Any}()
const V10_LOADED_CASE = Ref("")

function v10_gui_publish()
    s=V10_GUI_SESSION[];c=s.case;rows=s.rows
    recovery=isempty(rows) ? Dict[] : [Dict(string(k)=>(ismissing(v) ? nothing : v) for (k,v) in pairs(r)) for r in eachrow(v10_recovery(s))]
    identity=isempty(V10_LOADED_CASE[]) ? V10_GUI_ID[] : V10_SAVED_META[V10_LOADED_CASE[]].run_id
    global v10_gui_snapshot=Dict("run_id"=>identity,"executed_hours"=>s.next_hour,
        "parameters"=>Dict(string(k)=>v for (k,v) in pairs(c.config)),
        "rows"=>[Dict(string(k)=>v for (k,v) in pairs(row)) for row in rows],
        "recovery"=>recovery,"native_transition_calls"=>V10_NATIVE_RUNS[],
        "native_power_residual_gw"=>V10_NATIVE_MAX_POWER_RESIDUAL[],
        "elapsed_seconds"=>s.elapsed_seconds,"initial_fuel"=>V10_INITIAL_FUEL,
        "storage_energy"=>V10_STORAGE_ENERGY,"storage_power"=>c.storage_power,
        "backend"=>"AnyMath native resource transition + Julia JuMP/HiGHS MPC",
        "complete"=>s.next_hour==c.config.hours,"batch_running"=>V10_BATCH_RUNNING[],
        "loaded_record"=>!isempty(V10_LOADED_CASE[]),"saved_cases"=>sort(collect(keys(V10_SAVED_SESSIONS))))
    return nothing
end

function v10_gui_reset(;kwargs...)
    @assert !V10_BATCH_RUNNING[] "The verification batch is running; parameter experiments resume after it finishes."
    V10_LOADED_CASE[]=""
    V10_GUI_ID[]+=1
    V10_GUI_SESSION[]=v10_session(v10_config(;kwargs...);plant_backend="ANYMATH_NATIVE_BLOCK_FEEDBACK")
    V10_NATIVE_RUNS[]=0;V10_NATIVE_MAX_POWER_RESIDUAL[]=0.0
    v10_gui_publish()
end

function v10_gui_load_saved(name)
    @assert !V10_BATCH_RUNNING[] && haskey(V10_SAVED_SESSIONS,name)
    V10_GUI_SESSION[]=V10_SAVED_SESSIONS[name]
    metadata=V10_SAVED_META[name]
    V10_NATIVE_RUNS[]=metadata.native_calls;V10_NATIVE_MAX_POWER_RESIDUAL[]=metadata.power_residual
    V10_LOADED_CASE[]=name
    v10_gui_publish()
    return nothing
end

function v10_gui_restore_saved(directory)
    for name in ["base","mild","contract","power","decentralized"]
        execution=joinpath(directory,name*"_execution.toml")
        isfile(execution) || continue
        metadata=TOML.parsefile(execution)
        @assert metadata["status"]=="COMPLETED" && metadata["plant_backend"]=="ANYMATH_NATIVE_BLOCK_FEEDBACK"
        parameters=(; (Symbol(k)=>v for (k,v) in metadata["parameters"])...)
        session=v10_session(v10_config(;parameters...);plant_backend=metadata["plant_backend"])
        rows=CSV.read(joinpath(directory,name*"_actions_states.csv"),DataFrame)
        @assert nrow(rows)==session.case.config.hours && rows.hour==collect(0:nrow(rows)-1) && all(rows.lp_status.=="OPTIMAL")
        session.rows=NamedTuple[NamedTuple(row) for row in eachrow(rows)]
        session.audit_rows=NamedTuple[NamedTuple(row) for row in eachrow(CSV.read(joinpath(directory,name*"_physical_audit.csv"),DataFrame))]
        session.next_hour=nrow(rows)
        lastrow=last(session.rows)
        session.state=(fuel=[lastrow.coal_stock_mtce,lastrow.oil_stock_mtce,lastrow.gas_stock_mtce],
            soc=[lastrow.battery_soc_gwh,lastrow.pumped_soc_gwh],thermal=[lastrow.coal_gw,lastrow.gas_gw],debt=lastrow.shift_debt_gwh)
        session.elapsed_seconds=CSV.read(joinpath(directory,name*"_summary.csv"),DataFrame).elapsed_seconds[1]
        V10_SAVED_SESSIONS[name]=session
        V10_SAVED_META[name]=(run_id="已保存-"*name,native_calls=metadata["native_transition_calls"],power_residual=metadata["native_maximum_power_residual_gw"])
    end
    return nothing
end

function v10_gui_step(n=24)
    @assert !V10_BATCH_RUNNING[] "The verification batch owns the native model until it finishes."
    s=V10_GUI_SESSION[]
    @assert s.next_hour<s.case.config.hours "This run is complete. Start a new parameter experiment."
    step_v10!(s,n;plant_step=v10_native_step)
    v10_gui_publish()
    if s.next_hour==s.case.config.hours
        v10_export(s,joinpath(@__DIR__,"V10_results");name="interactive_"*lpad(string(V10_GUI_ID[]),3,'0'))
    end
    return nothing
end
