//! Game data from data/*.json, read at runtime (never hard-coded). Numbers are
//! parsed with Godot's own strtod so every value has the same bits as in GDScript.

use serde_json::Value;
use std::path::Path;

pub const RES_NAMES: [&str; 6] = ["coal", "wood", "parts", "raw_food", "rations", "cores"];
pub const NRES: usize = 6;
pub const DIFFICULTY_KEYS: [&str; 4] = ["GENTLE", "STANDARD", "HARSH", "HARDCORE"];

/// Godot 4.7.2 `built_in_strtod` (core/string/ustring.cpp), used by its JSON parser.
pub fn godot_strtod(s: &str) -> f64 {
    let b = s.as_bytes();
    let mut p = 0usize;
    while p < b.len() && (b[p] == b' ' || b[p] == b'\t' || b[p] == b'\n') {
        p += 1;
    }
    let mut sign = false;
    if p < b.len() && b[p] == b'-' {
        sign = true;
        p += 1;
    } else if p < b.len() && b[p] == b'+' {
        p += 1;
    }
    let at = |i: usize| -> u8 { if i < b.len() { b[i] } else { 0 } };
    let mut dec_pt: i64 = -1;
    let mut mant_size: i64 = 0;
    loop {
        let c = at(p);
        if !c.is_ascii_digit() {
            if c != b'.' || dec_pt >= 0 {
                break;
            }
            dec_pt = mant_size;
        }
        p += 1;
        mant_size += 1;
    }
    let p_exp = p;
    p -= mant_size as usize;
    if dec_pt < 0 {
        dec_pt = mant_size;
    } else {
        mant_size -= 1;
    }
    let frac_exp;
    if mant_size > 18 {
        frac_exp = dec_pt - 18;
        mant_size = 18;
    } else {
        frac_exp = dec_pt - mant_size;
    }
    let mut fraction: f64;
    if mant_size == 0 {
        fraction = 0.0;
        return if sign { -fraction } else { fraction };
    } else {
        let mut frac1: i32 = 0;
        while mant_size > 9 {
            let mut c = at(p);
            p += 1;
            if c == b'.' {
                c = at(p);
                p += 1;
            }
            frac1 = 10 * frac1 + (c - b'0') as i32;
            mant_size -= 1;
        }
        let mut frac2: i32 = 0;
        while mant_size > 0 {
            let mut c = at(p);
            p += 1;
            if c == b'.' {
                c = at(p);
                p += 1;
            }
            frac2 = 10 * frac2 + (c - b'0') as i32;
            mant_size -= 1;
        }
        fraction = (1.0e9 * frac1 as f64) + frac2 as f64;
    }
    p = p_exp;
    let mut exp: i64 = 0;
    let mut exp_sign = false;
    let mut have_exp = true;
    if at(p) == b'E' || at(p) == b'e' {
        p += 1;
        if at(p) == b'-' {
            exp_sign = true;
            p += 1;
        } else if at(p) == b'+' {
            p += 1;
        }
        if !at(p).is_ascii_digit() {
            have_exp = false;
        }
        while have_exp && at(p).is_ascii_digit() {
            exp = exp * 10 + (at(p) - b'0') as i64;
            p += 1;
        }
    }
    if !have_exp {
        exp = 0;
        exp_sign = false;
    }
    exp = if exp_sign { frac_exp - exp } else { frac_exp + exp };
    let neg = exp < 0;
    if neg {
        exp = -exp;
    }
    if exp > 511 {
        exp = 511;
    }
    const P10: [f64; 9] = [10., 100., 1.0e4, 1.0e8, 1.0e16, 1.0e32, 1.0e64, 1.0e128, 1.0e256];
    let mut dbl_exp = 1.0f64;
    let mut i = 0;
    while exp != 0 {
        if exp & 1 != 0 {
            dbl_exp *= P10[i];
        }
        exp >>= 1;
        i += 1;
    }
    if neg {
        fraction /= dbl_exp;
    } else {
        fraction *= dbl_exp;
    }
    if sign {
        -fraction
    } else {
        fraction
    }
}

pub fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => godot_strtod(&n.to_string()),
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => panic!("expected number, got {v}"),
    }
}

fn g<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("missing key {k}"))
}
fn gn(v: &Value, k: &str) -> f64 {
    num(g(v, k))
}
fn gs(v: &Value, k: &str) -> String {
    g(v, k).as_str().unwrap().to_string()
}
fn arr(v: &Value) -> &Vec<Value> {
    v.as_array().unwrap()
}
fn obj(v: &Value) -> &serde_json::Map<String, Value> {
    v.as_object().unwrap()
}

pub fn res_index(name: &str) -> usize {
    RES_NAMES.iter().position(|r| *r == name).unwrap_or_else(|| panic!("unknown resource {name}"))
}

fn res_list(v: &Value) -> Vec<(usize, f64)> {
    obj(v).iter().map(|(k, x)| (res_index(k), num(x))).collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shift {
    Day,
    Night,
    Always,
}

#[derive(Clone, Debug)]
pub struct BDef {
    pub id: String,
    pub size: [f64; 2],
    pub cost: Vec<(usize, f64)>,
    pub build_minutes: f64,
    pub insulation: f64,
    pub beds: f64,
    pub workers: f64,
    pub engineers_only: bool,
    pub researches: bool,
    pub min_level: Option<f64>,
    pub needs_coal_seam: bool,
    pub produces: Vec<(usize, f64)>,
    pub consumes: Vec<(usize, f64)>,
    pub has_consumes: bool,
    pub pile: Option<(usize, f64)>,
    pub rate: f64,
    pub gathers_piles_radius: Option<f64>,
    pub gather_rates: [f64; NRES],
    pub harvests_forest_radius: Option<f64>,
    pub shift: Shift,
    pub storm_stops: bool,
    pub care_beds: Option<f64>,
    pub cure_hours_sick: Option<f64>,
    pub cure_hours_grave: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct TechDef {
    pub id: String,
    pub tier: f64,
    pub requires: Vec<usize>,
    pub cost: Vec<(usize, f64)>,
    pub hours: f64,
    pub opens_tier: f64, // -1 when absent
    pub unlock: Vec<usize>,
    pub furnace_max: Option<f64>,
    pub heater_max: Option<f64>,
    pub heater_coal_minus: Option<f64>,
    pub furnace_coal_mult: Option<f64>,
    pub max_teams: Option<f64>,
    pub scout_speed: Option<f64>,
    pub cure_mult: Option<f64>,
    pub overdrive_levels: Option<f64>,
    pub pressure_up_mult: Option<f64>,
    pub pressure_down_mult: Option<f64>,
    pub rate_mult: Option<Vec<(usize, f64)>>,
    pub insulation: Option<Vec<(usize, f64)>>,
}

#[derive(Clone, Debug)]
pub struct SiteDef {
    pub id: String,
    pub hours: [f64; 2],
    pub loot: Vec<(usize, f64)>,
    pub people: Vec<(u8, f64)>,
    pub risky: bool,
}

#[derive(Clone, Debug)]
pub struct Storm {
    pub day: f64,
    pub temps: Vec<f64>,
    pub end_c: Option<f64>,
    pub final_phase: Option<(f64, f64, f64)>, // start_h, end_h, c
    pub scenario_end: bool,
}

/// balance weather.chaos (see sim/weather.gd); all integers in the sim.
#[derive(Clone, Debug, Default)]
pub struct Chaos {
    pub seed_xor: i64,
    pub sigma: i64,
    pub rho: i64,
    pub beta_num: i64,
    pub beta_den: i64,
    pub dt_div: i64,
    pub steps_per_hour: i64,
    pub spinup_steps: i64,
    pub first_day: i64,
    pub min_gap_days: i64,
    pub peak_c: i64,
    pub tier_shallow_below_c: i64,
    pub tier_deep_from_c: i64,
    pub sc_open_day: i64,
    pub sc_peak_c: i64,
    pub sc_force_day: i64,
    pub oracle_horizon_days: i64,
    /// observation noise of the selected level (noise_level), and both levels for resets
    pub p_noise: i64,
    pub w_noise: i64,
    pub murky: (i64, i64),
    pub clear: (i64, i64),
    pub p_base: i64,
    pub p_z_mid: i64,
    pub p_z_gain: i64,
    pub w_base: i64,
    pub w_y_div: i64,
    pub w_storm_base: i64,
    pub z1_c: i64,
    pub z2_c: i64,
    pub animal_noise_pct: i64,
    pub halo_calm_pct: i64,
    pub halo_s1_pct: i64,
    pub grey_s2_pct: i64,
    pub warn_trend: i64,
    pub warn_wind_ms: i64,
}

impl Chaos {
    /// RL reset option weather_noise ("murky" = the game, "clear" = experiments).
    pub fn set_noise(&mut self, level: &str) {
        let (p, w) = if level == "clear" { self.clear } else { self.murky };
        self.p_noise = p;
        self.w_noise = w;
    }
}

fn chaos_defs(c: &Value) -> Chaos {
    let i = |v: &Value, k: &str| gn(v, k) as i64;
    let lor = g(c, "lorenz");
    let (pc, wc, sc, wn, scn, nl) = (g(c, "pressure"), g(c, "wind"), g(c, "signs"), g(c, "warning"), g(c, "scenario"), g(c, "noise_levels"));
    let lvl = |k: &str| (i(g(nl, k), "pressure"), i(g(nl, k), "wind"));
    let mut ch = Chaos {
        seed_xor: i(c, "seed_xor"),
        sigma: i(lor, "sigma"),
        rho: i(lor, "rho"),
        beta_num: i(lor, "beta_num"),
        beta_den: i(lor, "beta_den"),
        dt_div: i(lor, "dt_div"),
        steps_per_hour: i(lor, "steps_per_hour"),
        spinup_steps: i(lor, "spinup_steps"),
        first_day: i(c, "first_day"),
        min_gap_days: i(c, "min_gap_days"),
        peak_c: i(c, "peak_c"),
        tier_shallow_below_c: i(c, "tier_shallow_below_c"),
        tier_deep_from_c: i(c, "tier_deep_from_c"),
        sc_open_day: i(scn, "open_day"),
        sc_peak_c: i(scn, "peak_c"),
        sc_force_day: i(scn, "force_day"),
        oracle_horizon_days: i(c, "oracle_horizon_days"),
        p_noise: 0,
        w_noise: 0,
        murky: lvl("murky"),
        clear: lvl("clear"),
        p_base: i(pc, "base"),
        p_z_mid: i(pc, "z_mid"),
        p_z_gain: i(pc, "z_gain"),
        w_base: i(wc, "base"),
        w_y_div: i(wc, "y_div"),
        w_storm_base: i(wc, "storm_base"),
        z1_c: i(sc, "z1_c"),
        z2_c: i(sc, "z2_c"),
        animal_noise_pct: i(sc, "animal_noise_pct"),
        halo_calm_pct: i(sc, "halo_calm_pct"),
        halo_s1_pct: i(sc, "halo_s1_pct"),
        grey_s2_pct: i(sc, "grey_s2_pct"),
        warn_trend: i(wn, "trend"),
        warn_wind_ms: i(wn, "wind_ms"),
    };
    let level = c.get("noise_level").and_then(|v| v.as_str()).unwrap_or("murky").to_string();
    ch.set_noise(&level);
    ch
}

#[derive(Clone, Debug)]
pub struct Profile {
    pub id: String,
    pub script: Vec<(f64, f64)>,
    pub storms: Vec<Storm>,
    pub random_min_c: f64,
    pub random_max_c: f64,
    pub random_change_chance: f64,
    pub rs_first_day: f64,
    pub rs_first_interval_days: f64,
    pub rs_interval_step_days: f64,
    pub rs_min_interval_days: f64,
    pub rs_days: f64,
    pub rs_temps: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct LawDef {
    pub id: String,
    pub hope_tiers: f64,
    pub discontent_tiers: f64,
    pub raw_per_batch: f64,
    pub rations_per_batch: f64,
    pub sick_chance: f64,
    pub jobs: Vec<usize>,
    pub start_hour: f64,
    pub end_hour: f64,
    pub discontent_per_building_day: f64,
}

#[derive(Clone, Debug)]
pub struct Wave {
    pub day: f64,
    pub all: f64,
    pub sick_all: f64,
    pub healthy: f64,
}

#[derive(Clone, Debug)]
pub struct Defs {
    pub buildings: Vec<BDef>,
    pub techs: Vec<TechDef>,
    pub sites: Vec<SiteDef>,
    pub profiles: Vec<Profile>,
    pub laws: Vec<LawDef>,
    pub law_soup: Option<usize>,
    pub law_additives: Option<usize>,
    pub law_child: Option<usize>,
    pub law_extended: Option<usize>,
    pub laws_points_per_tier: f64,
    pub k: f64,
    // time
    pub minutes_per_day: f64,
    pub work_start_hour: f64,
    pub work_end_hour: f64,
    pub hunt_start_hour: f64,
    pub hunt_end_hour: f64,
    pub meal_hour: f64,
    pub weather_change_hour: f64,
    pub sickness_roll_hour: f64,
    // map
    pub map_size: f64,
    pub forest_patches: f64,
    pub coal_piles: f64,
    pub wood_crates: f64,
    pub scrap_heaps: f64,
    pub ring_min: f64,
    pub ring_max: f64,
    pub forest_tile_amount: f64,
    pub coal_seams: f64,
    // heat
    pub zero_c: f64,
    pub c_per_level: f64,
    pub fire_spot_bonus: f64,
    pub furnace_coal_per_level_hour: f64,
    pub furnace_radius: f64,
    pub furnace_max_level: f64,
    pub relight_minutes: f64,
    pub od_levels: f64,
    pub od_up_h: f64,
    pub od_down_h: f64,
    pub od_max_pressure: f64,
    pub od_reset_pressure: f64,
    pub heater_coal_per_level_hour: f64,
    pub heater_building_scale: f64,
    pub heater_max_level: f64,
    pub forecast_days: f64,
    pub profile_by_difficulty: [usize; 4],
    pub weather_chaotic: bool, // balance weather.mode == "chaotic" (precursors, no dated forecast)
    pub chaos: Chaos,
    // body
    pub change_hours_by_diff: Vec<f64>,
    pub livable_to_comfortable_hours: f64,
    pub sick_target: [f64; 6], // levels -3..2
    pub sick_chance_approach_per_day: f64,
    pub gravely_ill_from_level: f64,
    pub gravely_ill_chance: f64,
    pub body_stage_days: f64,
    pub hypo_safe_c: f64,
    pub hypo_min_rate_h: f64,
    pub hypo_rate_per_c_h: f64,
    pub hypo_recover_factor_h: f64,
    pub hypo_recover_min_h: f64,
    pub hypo_recover_max_h: f64,
    pub hypo_recover_level: f64,
    pub hypo_stage_work: Vec<(f64, f64)>,
    pub hypo_stage_move: Vec<(f64, f64)>,
    pub warm_up_from: f64,
    pub warm_up_until: f64,
    pub frostbite_min_sev: f64,
    pub frostbite_chance_h: f64,
    pub frostbite_work_loss: f64,
    pub frostbite_fingers: i64,
    pub frostbite_digits: i64,
    pub walk_tiles_per_minute: f64,
    // food
    pub rations_per_day: f64,
    pub starving_rations: f64,
    pub food_stage_days: f64,
    pub raw_sick_chance: f64,
    // mood
    pub start_hope: f64,
    pub start_discontent: f64,
    pub death_hope: f64,
    pub death_discontent: f64,
    pub storm_forecast_hope: f64,
    pub needs_discontent_tiers: f64,
    pub daily_hope_drift: f64,
    pub daily_discontent_drift: f64,
    // population
    pub pop_floor: f64,
    pub wanderer_check_days: f64,
    pub wanderers_per_arrival: f64,
    pub wanderer_supplies: Vec<(usize, f64)>,
    // events
    pub waves: Vec<Wave>,
    pub role_worker: f64,
    pub role_engineer: f64,
    pub role_child: f64,
    // start
    pub start_roles: [f64; 3],
    pub start_resources: Vec<(usize, f64)>,
    // expeditions
    pub team_members: f64,
    pub team_allowed: [bool; 3],
    pub team_cost: Vec<(usize, f64)>,
    pub max_teams: f64,
    pub return_factor: f64,
    pub group_walk_factor: f64,
    pub risky_loss_chance: f64,
    // tech
    pub difficulty_hours_add: [f64; 4],
    pub efficiency_per_engineer: f64,
    pub speed_bands: Vec<(f64, f64)>,
    // gear (RW quality tiers)
    pub gear_insulation_mult: Vec<f64>,
    pub gear_tool_mult: Vec<f64>,
    pub gear_tool_work_bonus: f64,
    pub gear_kind_insulation: Vec<f64>,
    pub gear_table_normal: Vec<f64>,
    pub gear_table_risky: Vec<f64>,
    pub gear_long_site_hours: f64,
    pub gear_items_per_site: f64,
    // RW mental breaks / FP ultimatum
    pub breaks: Vec<(f64, f64, f64, f64)>, // below, mtb_days, hours lo, hours hi
    pub catharsis_days: f64,
    pub ult_discontent_at: f64,
    pub ult_hope_at: f64,
    pub ult_days: f64,
    pub ult_clear_discontent: f64,
    pub ult_clear_hope: f64,
    pub ult_success_hope_tiers: f64,
    // RW life cycle
    pub days_per_year: f64,
    pub child_rate: f64,
    pub child_rate_until: f64,
    pub adult_rate_from: f64,
    pub adult_age: i64,
    pub heart_attack_year: Vec<(f64, f64)>,
    pub heart_attack_death: f64,
    pub partner_min_age: i64,
    pub pregnancy_chance_night: f64,
    pub fertility_female: Vec<(f64, f64)>,
    pub fertility_male: Vec<(f64, f64)>,
    pub gestation_days: f64,
    pub birth_healthy: f64,
    pub birth_sick: f64,
    pub mother_death: Vec<(f64, f64)>,
    pub birth_quality_base: f64,
    pub birth_quality_bed: f64,
    pub baby_hope: f64,
    pub pregnancy_work: Vec<(f64, f64)>,
    pub child_work_age_default: i64,
    pub child_work_age: Vec<(usize, i64)>, // building def -> min age
    pub n_chapters: i64,
    // offline catch-up (GDD 7)
    pub off_game_days_per_real_day: f64,
    pub off_max_real_hours: f64,
}


fn opt(v: &Value, k: &str) -> Option<f64> {
    v.get(k).map(num)
}

fn role_index(n: &str) -> u8 {
    match n {
        "worker" => 0,
        "engineer" => 1,
        "child" => 2,
        _ => panic!("unknown role {n}"),
    }
}

fn pairs(v: &Value) -> Vec<(f64, f64)> {
    arr(v).iter().map(|e| (num(&arr(e)[0]), num(&arr(e)[1]))).collect()
}

impl Defs {
    pub fn load_dir(dir: &Path) -> Defs {
        let read = |n: &str| -> Value {
            let t = std::fs::read_to_string(dir.join(n)).unwrap_or_else(|e| panic!("missing data file {n}: {e}"));
            serde_json::from_str(&t).expect("invalid JSON")
        };
        Defs::from_values(&read("balance.json"), &read("buildings.json"), &read("expeditions.json"), &read("tech.json"))
    }

    pub fn from_values(bal: &Value, bjs: &Value, exp: &Value, tech: &Value) -> Defs {
        let blist = arr(g(bjs, "buildings"));
        let bids: Vec<String> = blist.iter().map(|b| gs(b, "id")).collect();
        let bix = |n: &str| bids.iter().position(|x| x == n).unwrap_or_else(|| panic!("unknown building {n}"));
        let mut buildings = Vec::new();
        for b in blist {
            let size = arr(g(b, "size"));
            let mut gr = [0.0; NRES];
            if let Some(m) = b.get("gather_rates") {
                for (k, x) in obj(m) {
                    gr[res_index(k)] = num(x);
                }
            }
            let shift = match b.get("shift").and_then(|s| s.as_str()).unwrap_or("day") {
                "day" => Shift::Day,
                "night" => Shift::Night,
                "always" => Shift::Always,
                s => panic!("shift {s}"),
            };
            let needs = b.get("needs_tile").and_then(|s| s.as_str()).unwrap_or("");
            buildings.push(BDef {
                id: gs(b, "id"),
                size: [num(&size[0]), num(&size[1])],
                cost: res_list(g(b, "cost")),
                build_minutes: gn(b, "build_minutes"),
                insulation: gn(b, "insulation"),
                beds: opt(b, "beds").unwrap_or(0.0),
                workers: opt(b, "workers").unwrap_or(0.0),
                engineers_only: b.get("engineers_only").and_then(|x| x.as_bool()).unwrap_or(false),
                researches: b.get("researches").and_then(|x| x.as_bool()).unwrap_or(false),
                min_level: opt(b, "min_level"),
                needs_coal_seam: needs == "coal_seam",
                produces: b.get("produces").map(res_list).unwrap_or_default(),
                consumes: b.get("consumes").map(res_list).unwrap_or_default(),
                has_consumes: b.get("consumes").is_some(),
                pile: b.get("pile").map(|p| (res_index(p.get("resource").unwrap().as_str().unwrap()), gn(p, "amount"))),
                rate: opt(b, "rate").unwrap_or(0.0),
                gathers_piles_radius: opt(b, "gathers_piles_radius"),
                gather_rates: gr,
                harvests_forest_radius: opt(b, "harvests_forest_radius"),
                shift,
                storm_stops: b.get("storm_stops").and_then(|x| x.as_bool()).unwrap_or(false),
                care_beds: opt(b, "care_beds"),
                cure_hours_sick: opt(b, "cure_hours_sick"),
                cure_hours_grave: opt(b, "cure_hours_grave"),
            });
        }
        let tlist = arr(g(tech, "techs"));
        let tids: Vec<String> = tlist.iter().map(|t| gs(t, "id")).collect();
        let tix = |n: &str| tids.iter().position(|x| x == n).unwrap_or_else(|| panic!("unknown tech {n}"));
        let mut techs = Vec::new();
        for t in tlist {
            let e = g(t, "effects");
            let bmap = |k: &str| e.get(k).map(|m| obj(m).iter().map(|(b, x)| (bix(b), num(x))).collect::<Vec<_>>());
            techs.push(TechDef {
                id: gs(t, "id"),
                tier: gn(t, "tier"),
                requires: arr(g(t, "requires")).iter().map(|r| tix(r.as_str().unwrap())).collect(),
                cost: res_list(g(t, "cost")),
                hours: gn(t, "hours"),
                opens_tier: opt(e, "opens_tier").unwrap_or(-1.0),
                unlock: e.get("unlock").map(|u| arr(u).iter().map(|x| bix(x.as_str().unwrap())).collect()).unwrap_or_default(),
                furnace_max: opt(e, "furnace_max"),
                heater_max: opt(e, "heater_max"),
                heater_coal_minus: opt(e, "heater_coal_minus"),
                furnace_coal_mult: opt(e, "furnace_coal_mult"),
                max_teams: opt(e, "max_teams"),
                scout_speed: opt(e, "scout_speed"),
                cure_mult: opt(e, "cure_mult"),
                overdrive_levels: opt(e, "overdrive_levels"),
                pressure_up_mult: opt(e, "pressure_up_mult"),
                pressure_down_mult: opt(e, "pressure_down_mult"),
                rate_mult: bmap("rate_mult"),
                insulation: bmap("insulation"),
            });
        }
        let sites = arr(g(exp, "sites"))
            .iter()
            .map(|s| {
                let h = arr(g(s, "hours"));
                SiteDef {
                    id: gs(s, "id"),
                    hours: [num(&h[0]), num(&h[1])],
                    loot: s.get("loot").map(res_list).unwrap_or_default(),
                    people: s.get("people").map(|p| obj(p).iter().map(|(k, x)| (role_index(k), num(x))).collect()).unwrap_or_default(),
                    risky: s.get("risky").and_then(|x| x.as_bool()).unwrap_or(false),
                }
            })
            .collect();
        let mut profiles = Vec::new();
        for (pid, p) in obj(g(bal, "weather_profiles")) {
            let a = g(p, "after");
            let r = g(a, "recurring_storms");
            profiles.push(Profile {
                id: pid.clone(),
                script: pairs(g(p, "script")),
                storms: p
                    .get("storms")
                    .map(|s| {
                        arr(s)
                            .iter()
                            .map(|st| Storm {
                                day: gn(st, "day"),
                                temps: arr(g(st, "temps_by_day")).iter().map(num).collect(),
                                end_c: opt(st, "end_c"),
                                final_phase: st.get("final_phase").map(|f| (gn(f, "start_h"), gn(f, "end_h"), gn(f, "c"))),
                                scenario_end: st.get("scenario_end").and_then(|x| x.as_bool()).unwrap_or(false),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                random_min_c: gn(a, "random_min_c"),
                random_max_c: gn(a, "random_max_c"),
                random_change_chance: gn(a, "random_change_chance"),
                rs_first_day: gn(r, "first_day"),
                rs_first_interval_days: gn(r, "first_interval_days"),
                rs_interval_step_days: gn(r, "interval_step_days"),
                rs_min_interval_days: gn(r, "min_interval_days"),
                rs_days: gn(r, "days"),
                rs_temps: arr(g(r, "temps_c")).iter().map(num).collect(),
            });
        }
        let pix = |n: &str| profiles.iter().position(|p| p.id == n).unwrap();
        let wb = g(bal, "weather");
        let pbd = g(wb, "profile_by_difficulty");
        let profile_by_difficulty = [0, 1, 2, 3].map(|i| pix(g(pbd, DIFFICULTY_KEYS[i]).as_str().unwrap()));
        let ld = g(bal, "laws");
        let mut laws = Vec::new();
        for (k, l) in obj(ld) {
            if !l.is_object() {
                continue;
            }
            laws.push(LawDef {
                id: k.clone(),
                hope_tiers: gn(l, "hope_tiers"),
                discontent_tiers: gn(l, "discontent_tiers"),
                raw_per_batch: opt(l, "raw_per_batch").unwrap_or(0.0),
                rations_per_batch: opt(l, "rations_per_batch").unwrap_or(0.0),
                sick_chance: opt(l, "sick_chance").unwrap_or(0.0),
                jobs: l.get("jobs").map(|j| arr(j).iter().map(|x| bix(x.as_str().unwrap())).collect()).unwrap_or_default(),
                start_hour: opt(l, "start_hour").unwrap_or(0.0),
                end_hour: opt(l, "end_hour").unwrap_or(0.0),
                discontent_per_building_day: opt(l, "discontent_per_building_day").unwrap_or(0.0),
            });
        }
        let lix = |n: &str| laws.iter().position(|l| l.id == n);
        let time = g(bal, "time");
        let map = g(bal, "map");
        let heat = g(bal, "heat");
        let fur = g(bal, "furnace");
        let htr = g(bal, "heater");
        let body = g(bal, "body");
        let food = g(bal, "food");
        let mood = g(bal, "mood");
        let gear = g(bal, "gear");
        let life = g(bal, "life");
        let pop = g(bal, "population");
        let ev = g(bal, "events");
        let start = g(bal, "start");
        let team = g(exp, "team");
        let st = g(body, "sick_target_by_level");
        let rr = g(ev, "refugee_roles");
        let sr = g(start, "roles");
        let dha = g(tech, "difficulty_hours_add");
        let mut allowed = [false; 3];
        for r in arr(g(team, "allowed_roles")) {
            allowed[role_index(r.as_str().unwrap()) as usize] = true;
        }
        let d = Defs {
            law_soup: lix("soup"),
            law_additives: lix("food_additives"),
            law_child: lix("child_labour_safe"),
            law_extended: lix("extended_shift"),
            laws_points_per_tier: gn(ld, "points_per_tier"),
            laws,
            buildings,
            techs,
            sites,
            profiles,
            k: gn(g(bal, "scale"), "k"),
            minutes_per_day: gn(time, "minutes_per_day"),
            work_start_hour: gn(time, "work_start_hour"),
            work_end_hour: gn(time, "work_end_hour"),
            hunt_start_hour: gn(time, "hunt_start_hour"),
            hunt_end_hour: gn(time, "hunt_end_hour"),
            meal_hour: gn(time, "meal_hour"),
            weather_change_hour: gn(time, "weather_change_hour"),
            sickness_roll_hour: gn(time, "sickness_roll_hour"),
            map_size: gn(map, "size"),
            forest_patches: gn(map, "forest_patches"),
            coal_piles: gn(map, "coal_piles"),
            wood_crates: gn(map, "wood_crates"),
            scrap_heaps: gn(map, "scrap_heaps"),
            ring_min: gn(map, "resource_ring_min"),
            ring_max: gn(map, "resource_ring_max"),
            forest_tile_amount: gn(map, "forest_tile_amount"),
            coal_seams: gn(map, "coal_seams"),
            zero_c: gn(heat, "zero_c"),
            c_per_level: gn(heat, "c_per_level"),
            fire_spot_bonus: gn(heat, "fire_spot_bonus"),
            furnace_coal_per_level_hour: gn(fur, "coal_per_level_hour"),
            furnace_radius: gn(fur, "radius"),
            furnace_max_level: gn(fur, "max_level"),
            relight_minutes: gn(fur, "relight_minutes"),
            od_levels: gn(g(fur, "overdrive"), "levels"),
            od_up_h: gn(g(fur, "overdrive"), "pressure_up_h"),
            od_down_h: gn(g(fur, "overdrive"), "pressure_down_h"),
            od_max_pressure: gn(g(fur, "overdrive"), "max_pressure"),
            od_reset_pressure: gn(g(fur, "overdrive"), "reset_pressure"),
            heater_coal_per_level_hour: gn(htr, "coal_per_level_hour"),
            heater_building_scale: gn(htr, "building_scale"),
            heater_max_level: gn(htr, "max_level"),
            forecast_days: gn(wb, "forecast_days"),
            profile_by_difficulty,
            weather_chaotic: wb.get("mode").and_then(|m| m.as_str()).unwrap_or("scripted") == "chaotic",
            chaos: wb.get("chaos").map(chaos_defs).unwrap_or_default(),
            change_hours_by_diff: arr(g(body, "change_hours_by_diff")).iter().map(num).collect(),
            livable_to_comfortable_hours: gn(body, "livable_to_comfortable_hours"),
            sick_target: [-3, -2, -1, 0, 1, 2].map(|l: i32| gn(st, &l.to_string())),
            sick_chance_approach_per_day: gn(body, "sick_chance_approach_per_day"),
            gravely_ill_from_level: gn(body, "gravely_ill_from_level"),
            gravely_ill_chance: gn(body, "gravely_ill_chance"),
            body_stage_days: gn(body, "stage_days"),
            hypo_safe_c: gn(body, "hypo_safe_c"),
            hypo_min_rate_h: gn(body, "hypo_min_rate_h"),
            hypo_rate_per_c_h: gn(body, "hypo_rate_per_c_h"),
            hypo_recover_factor_h: gn(body, "hypo_recover_factor_h"),
            hypo_recover_min_h: gn(body, "hypo_recover_min_h"),
            hypo_recover_max_h: gn(body, "hypo_recover_max_h"),
            hypo_recover_level: gn(body, "hypo_recover_level"),
            hypo_stage_work: pairs(g(body, "hypo_stage_work")),
            hypo_stage_move: pairs(g(body, "hypo_stage_move")),
            warm_up_from: gn(body, "warm_up_from"),
            warm_up_until: gn(body, "warm_up_until"),
            frostbite_min_sev: gn(body, "frostbite_min_sev"),
            frostbite_chance_h: gn(body, "frostbite_chance_h"),
            frostbite_work_loss: gn(body, "frostbite_work_loss"),
            frostbite_fingers: gn(body, "frostbite_fingers") as i64,
            frostbite_digits: gn(body, "frostbite_digits") as i64,
            walk_tiles_per_minute: gn(body, "walk_tiles_per_minute"),
            rations_per_day: gn(food, "rations_per_day"),
            starving_rations: gn(food, "starving_rations"),
            food_stage_days: gn(food, "stage_days"),
            raw_sick_chance: gn(food, "raw_sick_chance"),
            start_hope: gn(mood, "start_hope"),
            start_discontent: gn(mood, "start_discontent"),
            death_hope: gn(mood, "death_hope"),
            death_discontent: gn(mood, "death_discontent"),
            storm_forecast_hope: gn(mood, "storm_forecast_hope"),
            needs_discontent_tiers: gn(mood, "needs_discontent_tiers"),
            daily_hope_drift: gn(mood, "daily_hope_drift"),
            daily_discontent_drift: gn(mood, "daily_discontent_drift"),
            pop_floor: gn(pop, "floor"),
            wanderer_check_days: gn(pop, "wanderer_check_days"),
            wanderers_per_arrival: gn(pop, "wanderers_per_arrival"),
            wanderer_supplies: res_list(g(pop, "wanderer_supplies")),
            waves: arr(g(ev, "refugee_waves"))
                .iter()
                .map(|w| Wave { day: gn(w, "day"), all: gn(w, "all"), sick_all: gn(w, "sick_all"), healthy: gn(w, "healthy") })
                .collect(),
            role_worker: gn(rr, "worker"),
            role_engineer: gn(rr, "engineer"),
            role_child: gn(rr, "child"),
            start_roles: [gn(sr, "worker"), gn(sr, "engineer"), gn(sr, "child")],
            start_resources: res_list(g(start, "resources")),
            team_members: gn(team, "members"),
            team_allowed: allowed,
            team_cost: res_list(g(team, "cost")),
            max_teams: gn(team, "max_teams"),
            return_factor: gn(team, "return_factor"),
            group_walk_factor: gn(team, "group_walk_factor"),
            risky_loss_chance: gn(exp, "risky_loss_chance"),
            difficulty_hours_add: [0, 1, 2, 3].map(|i| gn(dha, DIFFICULTY_KEYS[i])),
            efficiency_per_engineer: gn(tech, "efficiency_per_engineer"),
            speed_bands: pairs(g(tech, "speed_bands")),
            gear_insulation_mult: arr(g(gear, "insulation_mult")).iter().map(num).collect(),
            gear_tool_mult: arr(g(gear, "tool_mult")).iter().map(num).collect(),
            gear_tool_work_bonus: gn(gear, "tool_work_bonus"),
            gear_kind_insulation: arr(g(gear, "kinds")).iter().map(|k| gn(k, "insulation_c")).collect(),
            gear_table_normal: arr(g(gear, "quality_table_normal")).iter().map(num).collect(),
            gear_table_risky: arr(g(gear, "quality_table_risky")).iter().map(num).collect(),
            gear_long_site_hours: gn(gear, "long_site_hours"),
            gear_items_per_site: gn(gear, "items_per_site"),
            breaks: arr(g(mood, "breaks"))
                .iter()
                .map(|t| {
                    let h = arr(g(t, "hours"));
                    (gn(t, "below"), gn(t, "mtb_days"), num(&h[0]), num(&h[1]))
                })
                .collect(),
            catharsis_days: gn(mood, "catharsis_days"),
            ult_discontent_at: gn(g(mood, "ultimatum"), "discontent_at"),
            ult_hope_at: gn(g(mood, "ultimatum"), "hope_at"),
            ult_days: gn(g(mood, "ultimatum"), "days"),
            ult_clear_discontent: gn(g(mood, "ultimatum"), "clear_discontent"),
            ult_clear_hope: gn(g(mood, "ultimatum"), "clear_hope"),
            ult_success_hope_tiers: gn(g(mood, "ultimatum"), "success_hope_tiers"),
            days_per_year: gn(life, "days_per_year"),
            child_rate: gn(life, "child_rate"),
            child_rate_until: gn(life, "child_rate_until"),
            adult_rate_from: gn(life, "adult_rate_from"),
            adult_age: gn(life, "adult_age") as i64,
            heart_attack_year: pairs(g(life, "heart_attack_year")),
            heart_attack_death: gn(life, "heart_attack_death"),
            partner_min_age: gn(life, "partner_min_age") as i64,
            pregnancy_chance_night: gn(life, "pregnancy_chance_night"),
            fertility_female: pairs(g(life, "fertility_female")),
            fertility_male: pairs(g(life, "fertility_male")),
            gestation_days: gn(life, "gestation_days"),
            birth_healthy: gn(life, "birth_healthy"),
            birth_sick: gn(life, "birth_sick"),
            mother_death: pairs(g(life, "mother_death")),
            birth_quality_base: gn(life, "birth_quality_base"),
            birth_quality_bed: gn(life, "birth_quality_bed"),
            baby_hope: gn(life, "baby_hope"),
            pregnancy_work: pairs(g(life, "pregnancy_work")),
            child_work_age_default: gn(g(life, "child_work_age"), "default") as i64,
            child_work_age: obj(g(life, "child_work_age")).iter().filter(|(k, _)| k.as_str() != "default").map(|(k, v)| (bix(k), num(v) as i64)).collect(),
            n_chapters: arr(g(life, "chapters")).len() as i64,
            off_game_days_per_real_day: gn(g(bal, "offline"), "game_days_per_real_day"),
            off_max_real_hours: gn(g(bal, "offline"), "max_real_hours"),
        };
        // The start resources define the resource dictionary order; it must be RES_NAMES.
        for (i, (r, _)) in d.start_resources.iter().enumerate() {
            assert_eq!(*r, i, "start resources must list {:?} in order", RES_NAMES);
        }
        d
    }

    pub fn bix(&self, id: &str) -> usize {
        self.buildings.iter().position(|b| b.id == id).unwrap_or_else(|| panic!("unknown building {id}"))
    }
    pub fn tix(&self, id: &str) -> usize {
        self.techs.iter().position(|t| t.id == id).unwrap_or_else(|| panic!("unknown tech {id}"))
    }
}
