//! Port of tools/bot_player.gd (the scripted bot) and its placement helpers.

use crate::weather::{maxf, minf};
use crate::world::*;

const RESEARCH_ORDER: [&str; 34] = [
    "beacon", "heaters", "hunters_gear", "sawmill", "faster_gathering",
    "drawing_boards", "coal_mining", "generator_power_1", "bunkhouse", "medical_post_upgrade",
    "more_scouts", "sawmill_upgrade", "drafting_machines", "generator_efficiency_1",
    "coal_mining_optimisation", "improved_heaters", "gathering_post_insulation", "cookhouse_insulation",
    "healthcare_insulation", "lighter_sleds", "mechanical_calculators", "generator_power_2", "infirmary", "house",
    "heater_efficiency", "efficient_gathering", "difference_engine", "advanced_heaters",
    "generator_efficiency_2", "coal_mining_rationalisation", "gathering_post_insulation_2",
    "additional_scouts", "automatic_prototyping", "generator_power_3",
];

/// Building/tech indices the bot names, resolved once per data set.
#[derive(Clone)]
pub struct Ids {
    pub workshop: usize,
    pub hunter_hut: usize,
    pub cookhouse: usize,
    pub beacon: usize,
    pub sawmill: usize,
    pub medical_post: usize,
    pub infirmary: usize,
    pub coal_mine: usize,
    pub gathering_post: usize,
    pub coal_pile: usize,
    pub wood_crates: usize,
    pub scrap_heap: usize,
    pub tent: usize,
    pub house: usize,
    pub bunkhouse: usize,
    pub research_order: Vec<usize>,
    /// Off by default (as BotPlayer.use_overdrive).
    pub use_overdrive: bool,
}

impl Ids {
    pub fn new(w: &World) -> Ids {
        let d = &w.defs;
        Ids {
            workshop: d.bix("workshop"),
            hunter_hut: d.bix("hunter_hut"),
            cookhouse: d.bix("cookhouse"),
            beacon: d.bix("beacon"),
            sawmill: d.bix("sawmill"),
            medical_post: d.bix("medical_post"),
            infirmary: d.bix("infirmary"),
            coal_mine: d.bix("coal_mine"),
            gathering_post: d.bix("gathering_post"),
            coal_pile: d.bix("coal_pile"),
            wood_crates: d.bix("wood_crates"),
            scrap_heap: d.bix("scrap_heap"),
            tent: d.bix("tent"),
            house: d.bix("house"),
            bunkhouse: d.bix("bunkhouse"),
            research_order: RESEARCH_ORDER.iter().map(|t| d.tix(t)).collect(),
            use_overdrive: false,
        }
    }
}

pub fn hourly(w: &mut World, ids: &Ids) {
    laws(w, ids);
    furnace(w);
    if ids.use_overdrive {
        overdrive(w);
    }
    heaters(w);
    scout(w);
    research(w, ids);
    build(w, ids);
    staff(w, ids);
    choices(w);
}

// ---------------------------------------------------------------- helpers

fn alive(w: &World) -> Vec<usize> {
    (0..w.survivors.len()).filter(|&i| w.survivors[i].alive() && w.survivors[i].job_id != SCOUT_JOB).collect()
}

fn count(w: &World, d: usize) -> i64 {
    w.buildings.iter().filter(|b| b.def == d).count() as i64
}

pub fn site_pending(w: &World) -> bool {
    w.buildings.iter().any(|b| !b.built)
}

fn beds(w: &World) -> i64 {
    let mut n = 0;
    for b in &w.buildings {
        n += w.defs.buildings[b.def].beds as i64;
    }
    n
}

#[inline]
fn k(w: &World) -> f64 {
    w.defs.k
}

// ---------------------------------------------------------------- laws

fn laws(w: &mut World, ids: &Ids) {
    if let Some(l) = w.defs.law_soup {
        w.sign_law(l);
    }
    if let Some(l) = w.defs.law_child {
        w.sign_law(l);
    }
    if w.res[1] < 8.0 && w.day() >= 1 {
        if let Some(l) = w.defs.law_extended {
            w.sign_law(l);
        }
    }
    if w.has_law(w.defs.law_extended) {
        let targets: Vec<i64> =
            w.buildings.iter().filter(|b| b.def == ids.wood_crates || b.def == ids.sawmill).map(|b| b.id).collect();
        for bid in targets {
            let on = w.res[1] < 15.0;
            w.set_extended_shift(bid, on);
        }
    }
}

// ---------------------------------------------------------------- furnace

fn furnace(w: &mut World) {
    let max_level = w.furnace_max_level();
    let per_level_day = w.defs.furnace_coal_per_level_hour * k(w) * 24.0;
    let f = w.furnace();
    if !w.furnace_lit && w.buildings[f].level == 0 {
        if w.res[0] >= 200.0 * k(w) || (w.weather_level() < 0.0 && w.res[0] >= 2.0) {
            w.set_furnace_level(1);
        }
        return;
    }
    let mut worst_home = f64::INFINITY;
    for b in &w.buildings {
        let d = &w.defs.buildings[b.def];
        if b.built && d.beds as i64 > 0 {
            worst_home = minf(worst_home, w.weather_level() + d.insulation);
        }
    }
    if worst_home == f64::INFINITY {
        worst_home = w.weather_level();
    }
    let mut want = ((-worst_home).ceil() as i64).clamp(1, max_level);
    if w.weather.in_storm(w.minute) || w.weather.warning_active(&w.defs, w.minute) {
        let storm_level = (w.weather.coldest_storm_temp() - w.defs.zero_c) / w.defs.c_per_level;
        want = ((-(storm_level + 1.0)).ceil() as i64).clamp(want, max_level);
    }
    while want > 1 && w.res[0] < per_level_day * want as f64 * 0.5 {
        want -= 1;
    }
    if w.buildings[f].level != want {
        w.set_furnace_level(want);
    }
}

fn heaters(w: &mut World) {
    for bi in 0..w.buildings.len() {
        let b = &w.buildings[bi];
        let d = &w.defs.buildings[b.def];
        if !b.built || d.min_level.is_none() || d.beds as i64 > 0 {
            continue;
        }
        let ml = d.min_level.unwrap();
        let base = w.building_level(bi) - (if w.heater_running(bi) { b.heater as f64 } else { 0.0 });
        let bid = b.id;
        w.set_heater(bid, if base < ml { 1 } else { 0 });
    }
}

fn coal_target(w: &World) -> f64 {
    let per_level_day = w.defs.furnace_coal_per_level_hour * k(w) * 24.0;
    let storm_days = w.weather.next_storm_temps.len() as f64;
    per_level_day * w.defs.furnace_max_level * storm_days
}

fn scout(w: &mut World) {
    if !w.scouting_available() || w.res[1] < 5.0 {
        return;
    }
    let mut best: i64 = -1;
    for i in 0..w.sites.len() {
        if w.sites[i].visited {
            continue;
        }
        let busy = w.teams.iter().any(|t| t.site == i as i64);
        if !busy && (best < 0 || w.sites[i].hours < w.sites[best as usize].hours) {
            best = i as i64;
        }
    }
    if best < 0 {
        return;
    }
    let round_trip = (w.sites[best as usize].hours * 1.5 * 60.0) as i64 + 120;
    if w.weather.next_storm_start - w.minute < round_trip && w.minute < w.weather.next_storm_end {
        return;
    }
    if w.weather.chaotic && w.weather.warning_active(&w.defs, w.minute) {
        return;
    }
    let workers: Vec<usize> = alive(w)
        .into_iter()
        .filter(|&i| {
            let s = &w.survivors[i];
            s.role == WORKER && s.illness == NONE && s.hypo < 0.2
        })
        .collect();
    let mut ids = Vec::new();
    for i in (0..workers.len()).rev() {
        if ids.len() < 2 {
            ids.push(w.survivors[workers[i]].id);
        }
    }
    if ids.len() == 2 {
        w.send_expedition(best, &ids);
    }
}

// ---------------------------------------------------------------- building

fn build(w: &mut World, ids: &Ids) {
    if site_pending(w) {
        return;
    }
    let people = alive(w);
    let pop = people.len() as i64;
    let sick = people.iter().filter(|&&i| w.survivors[i].illness != NONE).count();
    let grave = people.iter().filter(|&&i| w.survivors[i].illness == GRAVE).count();
    if count(w, ids.workshop) == 0 {
        near_furnace(w, ids.workshop);
    } else if count(w, ids.hunter_hut) * 12 < pop {
        near_furnace(w, ids.hunter_hut);
    } else if count(w, ids.cookhouse) == 0 {
        near_furnace(w, ids.cookhouse);
    } else if count(w, ids.beacon) == 0 && !w.building_locked(ids.beacon) {
        near_furnace(w, ids.beacon);
    } else if count(w, ids.sawmill) == 0 && !w.building_locked(ids.sawmill) {
        near_forest(w, ids.sawmill);
    } else if beds(w) < pop {
        if !upgrade_home(w, ids) {
            near_furnace(w, ids.tent);
        }
    } else if sick > 0 && count(w, ids.medical_post) == 0 {
        near_furnace(w, ids.medical_post);
    } else if grave > 0 && count(w, ids.infirmary) == 0 && w.res[5] >= 1.0 && !w.building_locked(ids.infirmary) {
        near_furnace(w, ids.infirmary);
    } else if count(w, ids.coal_mine) == 0 && w.res[5] >= 1.0 && !w.building_locked(ids.coal_mine) {
        on_seam(w, ids.coal_mine);
    } else if count(w, ids.gathering_post) == 0 && coal_piles_left(w, ids) > 0 {
        near_piles(w, ids.gathering_post, ids);
    } else {
        upgrade_home(w, ids);
    }
}

fn order_at(w: &mut World, d: usize, cell: Option<(i64, i64)>) -> bool {
    match cell {
        Some((x, y)) => w.order_build(d, x, y).is_some(),
        None => false,
    }
}

pub fn on_seam(w: &mut World, d: usize) -> bool {
    let c = cell_on_seam(w, d);
    order_at(w, d, c)
}

/// Placement searches (RLEnv masks with them too).
pub fn cell_on_seam(w: &World, d: usize) -> Option<(i64, i64)> {
    for y in 0..w.map_size {
        for x in 0..w.map_size {
            if w.can_place(d, x, y) {
                return Some((x, y));
            }
        }
    }
    None
}

fn upgrade_home(w: &mut World, ids: &Ids) -> bool {
    for target in [ids.house, ids.bunkhouse] {
        if w.building_locked(target) {
            continue;
        }
        let mut ok = true;
        for &(r, c) in &w.defs.buildings[target].cost {
            if w.res[r] < c * 1.5 {
                ok = false;
            }
        }
        if ok {
            return near_furnace(w, target);
        }
    }
    false
}

fn coal_piles_left(w: &World, ids: &Ids) -> i64 {
    w.buildings.iter().filter(|b| b.def == ids.coal_pile && b.amount > 0.0).count() as i64
}

pub fn near_furnace(w: &mut World, d: usize) -> bool {
    let c = cell_near_furnace(w, d);
    order_at(w, d, c)
}

pub fn cell_near_furnace(w: &World, d: usize) -> Option<(i64, i64)> {
    let c = w.center(w.furnace());
    let (cx, cy) = (c.x as f64 as i64, c.y as f64 as i64);
    for r in 2..12i64 {
        for y in (cy - r)..(cy + r + 1) {
            for x in (cx - r)..(cx + r + 1) {
                if (x - cx).abs().max((y - cy).abs()) != r {
                    continue;
                }
                if w.can_place(d, x, y) {
                    return Some((x, y));
                }
            }
        }
    }
    None
}

pub fn near_piles(w: &mut World, d: usize, ids: &Ids) -> bool {
    let c = cell_near_piles(w, d, ids);
    order_at(w, d, c)
}

pub fn cell_near_piles(w: &World, d: usize, ids: &Ids) -> Option<(i64, i64)> {
    let mut best = (-1i64, -1i64);
    let mut best_n = 0i64;
    let radius = w.defs.buildings[d].gathers_piles_radius.unwrap();
    let piles: Vec<V2> = (0..w.buildings.len())
        .filter(|&bi| w.buildings[bi].def == ids.coal_pile && w.buildings[bi].amount > 0.0)
        .map(|bi| w.center(bi))
        .collect();
    for y in 0..w.map_size {
        for x in 0..w.map_size {
            if !w.can_place(d, x, y) {
                continue;
            }
            let p = V2::f(x as f64 + 0.5, y as f64 + 0.5);
            let mut n = 0i64;
            for c in &piles {
                if (c.distance_to(p) as f64) <= radius {
                    n += 1;
                }
            }
            if n > best_n {
                best_n = n;
                best = (x, y);
            }
        }
    }
    if best_n > 0 { Some(best) } else { None }
}

pub fn near_forest(w: &mut World, d: usize) -> bool {
    let c = cell_near_forest(w, d);
    order_at(w, d, c)
}

pub fn cell_near_forest(w: &World, d: usize) -> Option<(i64, i64)> {
    let radius = w.defs.buildings[d].harvests_forest_radius.unwrap();
    let c = w.center(w.furnace());
    let mut best = (-1i64, -1i64);
    let mut best_d = f64::INFINITY;
    for y in 0..w.map_size {
        for x in 0..w.map_size {
            if !w.can_place(d, x, y) {
                continue;
            }
            let mut near = false;
            'outer: for dy in -3..4i64 {
                for dx in -3..4i64 {
                    if w.tile_at(x + dx, y + dy) == FOREST && (V2::f(dx as f64, dy as f64).length() as f64) <= radius {
                        near = true;
                        break 'outer;
                    }
                }
            }
            let dd = V2::f(x as f64, y as f64).distance_to(c) as f64;
            if near && dd < best_d {
                best_d = dd;
                best = (x, y);
            }
        }
    }
    if best.0 >= 0 { Some(best) } else { None }
}

// ---------------------------------------------------------------- staffing

struct Plan {
    job: i64,
    n: i64,
    kids: bool,
}

fn staff(w: &mut World, ids: &Ids) {
    let people = alive(w);
    let pop = people.len() as i64;
    let mut engineers: Vec<i64> = Vec::new();
    let mut labour: Vec<i64> = Vec::new();
    let mut children: Vec<i64> = Vec::new();
    for &si in &people {
        let s = &w.survivors[si];
        match s.role {
            ENGINEER => engineers.push(s.id),
            WORKER => labour.push(s.id),
            _ => children.push(s.id),
        }
    }
    if let Some(med) = first_built(w, ids.medical_post) {
        let mid = w.buildings[med].id;
        let n = 2.min(engineers.len());
        for i in 0..n {
            w.assign_job(engineers[i], mid);
        }
        engineers = engineers[n..].to_vec();
    }
    if let Some(inf) = first_built(w, ids.infirmary) {
        let iid = w.buildings[inf].id;
        let n = 4.min(engineers.len());
        for i in 0..n {
            w.assign_job(engineers[i], iid);
        }
        engineers = engineers[n..].to_vec();
    }
    if let Some(ws) = first_built(w, ids.workshop) {
        let wid = w.buildings[ws].id;
        let n = 1.min(engineers.len());
        for i in 0..n {
            w.assign_job(engineers[i], wid);
        }
        engineers = engineers[n..].to_vec();
    }
    labour.extend_from_slice(&engineers);

    let mut plan: Vec<Plan> = Vec::new();
    let cook = first_built(w, ids.cookhouse);
    if let Some(c) = cook {
        plan.push(Plan { job: w.buildings[c].id, n: 1, kids: true });
    }
    let raw_per_ration = if cook.is_some() { 1.0 / (2.0 * w.ration_multiplier()) } else { 1.0 };
    let mut hunters = (pop as f64 * raw_per_ration).ceil() as i64;
    if cook.is_none() {
        hunters = hunters.min(labour.len() as i64 - 3);
    }
    for bi in all_built(w, ids.hunter_hut) {
        let n = hunters.min(6);
        plan.push(Plan { job: w.buildings[bi].id, n, kids: false });
        hunters -= n;
    }
    let lv = w.buildings[w.furnace()].level.max(1);
    let coal_day = lv as f64 * w.defs.furnace_coal_per_level_hour * k(w) * 24.0;
    let mut slots = coal_slots(w, ids);
    take_slots(&mut plan, &mut slots, coal_day);
    if count(w, ids.beacon) == 0 || w.res[2] < 4.0 {
        if let Some(b) = w.buildings.iter().find(|b| b.def == ids.scrap_heap) {
            plan.push(Plan { job: b.id, n: 2, kids: true });
        }
    }
    if site_pending(w) {
        plan.push(Plan { job: BUILDER_JOB, n: 1, kids: false });
    }
    let saw = first_built(w, ids.sawmill);
    if let Some(s) = saw {
        plan.push(Plan { job: w.buildings[s].id, n: 2, kids: false });
    } else if let Some(b) = w.buildings.iter().find(|b| b.def == ids.wood_crates) {
        plan.push(Plan { job: b.id, n: 2, kids: true });
    }
    if w.res[0] < coal_target(w) {
        take_slots(&mut plan, &mut slots, 1000.0);
    } else if let Some(s) = saw {
        plan.push(Plan { job: w.buildings[s].id, n: 2, kids: false });
    }
    for p in plan.iter_mut() {
        if p.kids {
            while p.n > 0 && !children.is_empty() {
                if w.assign_job(children[0], p.job) {
                    p.n -= 1;
                }
                children.remove(0);
            }
        }
    }
    for &c in &children {
        w.assign_job(c, -1);
    }
    let mut idx = 0usize;
    for p in &plan {
        for _ in 0..p.n.max(0) {
            if idx < labour.len() {
                w.assign_job(labour[idx], p.job);
                idx += 1;
            }
        }
    }
    while idx < labour.len() {
        let job = if site_pending(w) { BUILDER_JOB } else { -1 };
        w.assign_job(labour[idx], job);
        idx += 1;
    }
}

struct Slot {
    job: i64,
    per: f64,
    cap: i64,
    kids: bool,
}

fn coal_slots(w: &World, ids: &Ids) -> Vec<Slot> {
    let mut out = Vec::new();
    for bi in all_built(w, ids.coal_mine) {
        out.push(Slot { job: w.buildings[bi].id, per: 24.0, cap: 4, kids: false });
    }
    if let Some(post) = first_built(w, ids.gathering_post) {
        if w.nearest_pile(post, w.defs.buildings[ids.gathering_post].gathers_piles_radius.unwrap()).is_some() {
            out.push(Slot { job: w.buildings[post].id, per: 14.0, cap: 4, kids: true });
        }
    }
    for b in &w.buildings {
        if b.def == ids.coal_pile {
            out.push(Slot { job: b.id, per: 10.2, cap: 6, kids: true });
        }
    }
    out
}

fn take_slots(plan: &mut Vec<Plan>, slots: &mut [Slot], mut per_day: f64) {
    for sl in slots.iter_mut() {
        if per_day <= 0.0 || sl.cap <= 0 {
            continue;
        }
        let n = sl.cap.min((per_day / sl.per).ceil() as i64);
        plan.push(Plan { job: sl.job, n, kids: sl.kids });
        sl.cap -= n;
        per_day -= n as f64 * sl.per;
    }
}

fn first_built(w: &World, d: usize) -> Option<usize> {
    w.buildings.iter().position(|b| b.def == d && b.built)
}

fn all_built(w: &World, d: usize) -> Vec<usize> {
    (0..w.buildings.len()).filter(|&i| w.buildings[i].def == d && w.buildings[i].built).collect()
}

fn choices(w: &mut World) {
    while !w.pending.is_empty() {
        if w.pending[0].kind == 1 {
            let o = if w.pending[0].options.contains(&OPT_CORE) { OPT_CORE } else { OPT_CHILD };
            w.resolve_choice(0, o);
            continue;
        }
        w.resolve_choice(0, 0);
    }
}

/// Overdrive through storms and deep cold (<= -60), never past 90 % pressure.
fn overdrive(w: &mut World) {
    let cold = w.weather.in_storm(w.minute) || w.outdoor_temp() <= -60.0;
    let on = cold && w.pressure < 90.0;
    w.set_overdrive(on);
}

fn research(w: &mut World, ids: &Ids) {
    if w.research_id.is_some() {
        return;
    }
    for &t in &ids.research_order {
        if !w.has_tech(t) {
            w.start_research(t);
            return;
        }
    }
}

#[allow(dead_code)]
fn _unused(a: f64, b: f64) -> f64 {
    maxf(a, b)
}
