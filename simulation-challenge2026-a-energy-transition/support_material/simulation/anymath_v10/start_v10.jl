# AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
# First run prepares the native solver and opens the AnyMath application.
# Later experiments use the application's reset/solve button, not re-inclusion.
if Base.find_package("JuMP")===nothing || Base.find_package("HiGHS")===nothing
    import Pkg
    Pkg.add(["JuMP","HiGHS"])
end
if !isdefined(Main,:V10Session)
    include(joinpath(@__DIR__,"v10_dispatch.jl"))
end
if !isdefined(Main,:V10_NATIVE_MODEL)
    include(joinpath(@__DIR__,"v10_native_plant.jl"))
end
v10_build_native()
if !isdefined(Main,:V10_GUI_SESSION)
    include(joinpath(@__DIR__,"v10_gui.jl"))
end
v10_gui_reset()
v10_gui_restore_saved(normpath(joinpath(@__DIR__,"../../outputs/v10/anymath/cloud_final")))
if haskey(V10_SAVED_SESSIONS,"base")
    v10_gui_load_saved("base")
end
app_dir=joinpath(@__DIR__,"V10_Lab")
mkpath(app_dir)
cp(joinpath(@__DIR__,"v10_app.jl"),joinpath(app_dir,"app.jl");force=true)
cp(joinpath(@__DIR__,"v10_dashboard.html"),joinpath(app_dir,"v10_dashboard.html");force=true)
if isdefined(Main,:v10_application)
    engee.genie.stop(app_dir)
end
global v10_application=engee.genie.start(app_dir;devel=true,open_url=true,log_file=joinpath(@__DIR__,"v10_app_log.txt"))
println(v10_application)
