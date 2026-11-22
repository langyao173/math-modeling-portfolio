# AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
# Frozen annual knee state, 2060; hourly profiles are engineering designs.
const V10_GENERATORS = ["coal", "gas", "nuclear", "hydro", "wind", "solar", "biomass"]
const V10_FUELS = ["coal", "oil", "gas"]
const V10_STORES = ["battery", "pumped"]
const V10_INITIAL_FUEL = [390.91068493150624, 151.70346796408808, 274.47012654064974]
const V10_INITIAL_SOC = [122.97342803441325, 287.3457234861376]
const V10_STORAGE_POWER = [61.48671401720662, 95.78190782871253]
const V10_STORAGE_ENERGY = [245.9468560688265, 574.6914469722752]
const V10_ETA = [0.9486832980505138, 0.8831760866327847]
const V10_EFFICIENCY = [0.4, 0.55]
const V10_THERMAL_CAPACITY = [1768.9724760494498, 352.68821631519694]
const V10_IMPORT_CAPACITY_PER_H = [0.11065267447441188, 0.24534432134707918, 0.06297995788788871]
const V10_GENERATION_COST = [0.32, 0.65, 0.09, 0.025, 0.0, 0.0, 0.35]
const V10_FUEL_VALUE = [1042.048, 3500.0, 2910.4075000000007]
const V10_CARBON_PRICE = 668.1022774978924
const V10_FUEL_EMISSION = [2.77249896, 2.14824708, 1.6441563600000002]

"""A parameter change starts a fresh case; it never relabels an old trajectory.
The 360h demo is a separate, shorter engineering design, not the V6 2160h cohort.
"""
function v10_config(; design="demo", hours=(design=="demo" ? 360 : 2160),
    event_starts=(design=="demo" ? [48,192] : [120,1200]),
    event_duration_hours=(design=="demo" ? 48 : 480), shock_severity=1.0,
    supply_policy="inherited", contract_multiplier=1.0,
    contract_lead_hours=(design=="demo" ? 12 : 48), storage_power_multiplier=1.0,
    strategy="coordinated")
    return (; design, hours, event_starts=Int.(event_starts), event_duration_hours,
        shock_severity, supply_policy, contract_multiplier, contract_lead_hours,
        storage_power_multiplier, strategy)
end
