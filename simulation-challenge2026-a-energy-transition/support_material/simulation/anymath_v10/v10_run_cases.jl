# AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
# Sequential native-feedback experiments. The single native model must have one
# writer; the application refuses reset/step requests while this batch flag is on.
using SHA, TOML, Dates

if !isdefined(Main,:V10_BATCH_RUNNING)
    const V10_BATCH_RUNNING=Ref(false)
end

function v10_batch_source_hashes()
    files=["v10_dispatch.jl","v10_config.jl","v10_native_plant.jl","v10_gui.jl",
        "v10_app.jl","v10_dashboard.html","start_v10.jl","v10_run_cases.jl","v10_base_inputs.csv"]
    return Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in files)
end

function v10_batch_versions()
    versions=Dict("Julia"=>string(VERSION))
    for (identity,loaded) in Base.loaded_modules
        if identity.name in ["JuMP","HiGHS","CSV","DataFrames","Engee","MathOptInterface"]
            version=Base.pkgversion(loaded)
            versions[identity.name]=version===nothing ? "not exposed by loaded package" : string(version)
        end
    end
    return versions
end

function v10_batch_publish(case_name,index,count)
    v10_gui_publish()
    v10_gui_snapshot["batch_running"]=V10_BATCH_RUNNING[]
    v10_gui_snapshot["batch_case"]=case_name
    v10_gui_snapshot["batch_case_index"]=index
    v10_gui_snapshot["batch_case_count"]=count
    return nothing
end

"""Five actual native-feedback experiments, with one reoptimized LP per hour.
An unchanged, native base prefix may continue; all other parameter cases start
from the common initial state. The 360h design has two 48h shocks, h48 and h192.
"""
function v10_run_cases!(;output_dir="/user/V10_results",resume_base=true)
    @assert !V10_BATCH_RUNNING[] "The shared native model already has an active batch"
    mkpath(output_dir)
    cases=[("base",v10_config()),("mild",v10_config(shock_severity=.5)),
        ("contract",v10_config(supply_policy="contract")),
        ("power",v10_config(storage_power_multiplier=1.5)),
        ("decentralized",v10_config(strategy="decentralized"))]
    sources=v10_batch_source_hashes();versions=v10_batch_versions()
    V10_BATCH_RUNNING[]=true
    summaries=DataFrame[];recoveries=DataFrame[];records=Dict[]
    batch_started=time();batch_timestamp=string(Dates.now(Dates.UTC))*"Z"
    current_name="not_started";current_index=0
    try
        for (index,(name,config)) in enumerate(cases)
            current_name=name;current_index=index;V10_LOADED_CASE[]=""
            existing=V10_GUI_SESSION[]
            resume=index==1 && resume_base && existing!==nothing && existing.case.config==config &&
                existing.plant_backend=="ANYMATH_NATIVE_BLOCK_FEEDBACK" && existing.next_hour>0
            if !resume
                V10_GUI_ID[]+=1
                V10_GUI_SESSION[]=v10_session(config;plant_backend="ANYMATH_NATIVE_BLOCK_FEEDBACK")
                V10_NATIVE_RUNS[]=0;V10_NATIVE_MAX_POWER_RESIDUAL[]=0.0
            end
            session=V10_GUI_SESSION[]
            @assert session.plant_backend=="ANYMATH_NATIVE_BLOCK_FEEDBACK"
            prefix=session.next_hour;calls_before=V10_NATIVE_RUNS[]
            @assert calls_before==prefix "Native call counter must identify this exact case prefix"
            started=time();timestamp=string(Dates.now(Dates.UTC))*"Z"
            v10_batch_publish(name,index,length(cases))
            println("V10 ",name," starting at ",prefix,"/",config.hours," actual hours");flush(stdout)
            while session.next_hour<config.hours
                step_v10!(session,min(6,config.hours-session.next_hour);plant_step=v10_native_step)
                v10_batch_publish(name,index,length(cases))
                println("V10 ",name," ",session.next_hour,"/",config.hours,
                    " native calls=",V10_NATIVE_RUNS[]," ENS=",sum(row.unserved_gw for row in session.rows)," GWh")
                flush(stdout);yield()
            end
            @assert V10_NATIVE_RUNS[]==config.hours
            v10_export(session,output_dir;name)
            V10_SAVED_SESSIONS[name]=session
            V10_SAVED_META[name]=(run_id=V10_GUI_ID[],native_calls=V10_NATIVE_RUNS[],
                power_residual=V10_NATIVE_MAX_POWER_RESIDUAL[])
            summary=v10_summary(session);insertcols!(summary,1,:case_name=>name);push!(summaries,summary)
            recovery=v10_recovery(session);insertcols!(recovery,1,:case_name=>name);push!(recoveries,recovery)
            audit=DataFrame(session.audit_rows)
            record=Dict("case_name"=>name,"status"=>"COMPLETED",
                "started_utc"=>timestamp,"finished_utc"=>string(Dates.now(Dates.UTC))*"Z",
                "case_wall_seconds_this_batch"=>time()-started,
                "resumed_native_prefix_hours"=>prefix,"new_native_transition_calls"=>V10_NATIVE_RUNS[]-calls_before,
                "native_transition_calls"=>V10_NATIVE_RUNS[],"executed_hours"=>session.next_hour,
                "optimal_lp_hours"=>count(row.lp_status=="OPTIMAL" for row in session.rows),
                "maximum_physical_residual"=>maximum(Matrix(audit[:,Not(:hour)])),
                "native_maximum_power_residual_gw"=>V10_NATIVE_MAX_POWER_RESIDUAL[],
                "plant_backend"=>session.plant_backend,
                "information_contract"=>"COMMON_24H_PERFECT_FORECAST_ROLLING_FIRST_ACTION",
                "parameters"=>Dict(string(k)=>v for (k,v) in pairs(config)),
                "source_sha256"=>sources,"package_versions"=>versions)
            open(joinpath(output_dir,name*"_execution.toml"),"w") do io;TOML.print(io,record);end
            push!(records,record)
            CSV.write(joinpath(output_dir,"batch_summary.csv"),vcat(summaries...;cols=:union))
            CSV.write(joinpath(output_dir,"batch_recovery.csv"),vcat(recoveries...;cols=:union))
        end
        receipt=Dict("status"=>"COMPLETED_FIVE_SEQUENTIAL_NATIVE_FEEDBACK_CASES",
            "started_utc"=>batch_timestamp,"finished_utc"=>string(Dates.now(Dates.UTC))*"Z",
            "wall_seconds"=>time()-batch_started,"case_count"=>length(records),
            "native_transition_calls_total"=>sum(record["native_transition_calls"] for record in records),
            "new_native_transition_calls_total"=>sum(record["new_native_transition_calls"] for record in records),
            "resumed_native_prefix_hours_total"=>sum(record["resumed_native_prefix_hours"] for record in records),
            "single_native_model_sequential"=>true,"cases"=>records,
            "source_sha256"=>sources,"package_versions"=>versions)
        open(joinpath(output_dir,"batch_execution.toml"),"w") do io;TOML.print(io,receipt);end
        println("V10 batch complete: ",length(records)," cases, ",receipt["native_transition_calls_total"]," native hourly transitions");flush(stdout)
    catch failure
        session=V10_GUI_SESSION[]
        if session!==nothing && session.next_hour>0
            v10_export(session,output_dir;name=current_name*"_failed_prefix")
        end
        receipt=Dict("status"=>"FAILED","case_name"=>current_name,"case_index"=>current_index,
            "completed_case_count"=>length(records),"error"=>sprint(showerror,failure,catch_backtrace()),
            "source_sha256"=>sources,"package_versions"=>versions)
        open(joinpath(output_dir,"batch_failure.toml"),"w") do io;TOML.print(io,receipt);end
        rethrow(failure)
    finally
        V10_BATCH_RUNNING[]=false
        v10_batch_publish(current_name,current_index,length(cases))
    end
    return nothing
end
