# AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
module App
using GenieFramework, Engee
using Genie.Router, Genie.Requests, Genie.Renderer.Html, Genie.Renderer.Json
@genietools

snapshot() = engee.genie.recv("v10_gui_snapshot";context=Main)

route("/") do
    Genie.Renderer.respond(read(joinpath(@__DIR__,"v10_dashboard.html"),String), :html)
end

route("/api/status") do
    Genie.Renderer.Json.json(snapshot())
end

route("/api/reset",method=POST) do
    p=Genie.Requests.jsonpayload()
    severity=Float64(p["shock_severity"]);duration=Int(p["event_duration_hours"])
    power=Float64(p["storage_power_multiplier"]);contract=Float64(p["contract_multiplier"])
    policy=String(p["supply_policy"]);strategy=String(p["strategy"])
    @assert 0<=severity<=1 && 12<=duration<=72 && .5<=power<=2 && 0<=contract<=4
    @assert policy in ["inherited","contract"] && strategy in ["coordinated","decentralized"]
    engee.genie.eval("v10_gui_reset(;shock_severity=$severity,event_duration_hours=$duration,storage_power_multiplier=$power,contract_multiplier=$contract,supply_policy=$(repr(policy)),strategy=$(repr(strategy)))")
    Genie.Renderer.Json.json(snapshot())
end

route("/api/step",method=POST) do
    p=Genie.Requests.jsonpayload();n=Int(p["hours"]);@assert 1<=n<=6
    engee.genie.eval("v10_gui_step($n)")
    Genie.Renderer.Json.json(snapshot())
end

route("/api/load",method=POST) do
    p=Genie.Requests.jsonpayload();name=String(p["case_name"])
    @assert name in ["base","mild","contract","power","decentralized"]
    engee.genie.eval("v10_gui_load_saved($(repr(name)))")
    Genie.Renderer.Json.json(snapshot())
end
end

