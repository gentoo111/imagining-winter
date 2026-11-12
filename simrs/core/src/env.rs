//! Port of tools/rl_server.gd: action list, masks, observation, reward, reset
//! options (perturb, warm, no_floor, difficulty). Values are the raw f64 that
//! GDScript computes; `json_f64`/`obs_f32` give what Python sees after the
//! server's JSON.stringify round trip.

use crate::bot::{self, Ids};
use crate::defs::{Defs, DIFFICULTY_KEYS};
use crate::rng::Rng;
use crate::weather::{clampf, maxf, minf};
use crate::world::*;

pub const MAX_ACTIONS: i64 = 6;
pub const MAX_DAYS: i64 = 60;
pub const ACTION_COST: f64 = 0.002;
pub const RES_OBS: [usize; 6] = [0, 1, 2, 3, 4, 5]; // coal wood parts raw_food rations cores

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Wait,
    Bot,
    Furnace(i64),
    Overdrive(bool),
    Build(usize),
    Add(usize),
    Remove(usize),
    AddBuilder,
    RemoveBuilder,
    Research(usize),
    Law(usize),
    Scout,
    Choice(usize),
}

#[derive(Clone, Debug, Default)]
pub struct ResetOpts {
    pub seed: i64,
    pub perturb: bool,
    pub warm: f64,
    pub no_floor: bool,
    pub difficulty: String,
    /// "scripted" (default, FP dated storms: training and the existing traces) or "chaotic"
    pub weather: String,
    /// append the visible weather precursors to the observation
    pub precursor_obs: bool,
    /// chaotic weather only: append the true next landing (hours / 240) and its tier (experiments)
    pub oracle_obs: bool,
    /// chaotic weather observation noise: "murky" (the game; also when empty) or "clear"
    pub weather_noise: String,
    /// episode length cap in days (0 = MAX_DAYS)
    pub max_days: i64,
}

#[derive(Clone, Debug)]
pub struct StepOut {
    pub obs: Vec<f64>,
    pub mask: Vec<bool>,
    pub reward: f64,
    pub done: bool,
    pub day: i64,
    pub alive: i64,
}

#[derive(Clone)]
pub struct Env {
    pub base: Defs,
    pub w: World,
    pub ids: Ids,
    pub actions: Vec<Kind>,
    pub names: Vec<String>,
    pub build_ids: Vec<usize>,
    pub job_ids: Vec<usize>,
    pub actions_this_hour: i64,
    pub precursor_obs: bool,
    pub oracle_obs: bool,
    pub max_days: i64,
}

impl Env {
    pub fn new(base: Defs) -> Env {
        let mut build_ids = Vec::new();
        let mut job_ids = Vec::new();
        for (i, d) in base.buildings.iter().enumerate() {
            if d.id != "furnace" && d.pile.is_none() {
                build_ids.push(i);
            }
            if d.workers as i64 > 0 {
                job_ids.push(i);
            }
        }
        let mut actions = vec![Kind::Wait, Kind::Bot];
        let mut names = vec!["wait:".to_string(), "bot:".to_string()];
        for lv in 0..5 {
            actions.push(Kind::Furnace(lv));
            names.push(format!("furnace:{lv}"));
        }
        actions.push(Kind::Overdrive(true));
        names.push("overdrive:on".into());
        actions.push(Kind::Overdrive(false));
        names.push("overdrive:off".into());
        for &b in &build_ids {
            actions.push(Kind::Build(b));
            names.push(format!("build:{}", base.buildings[b].id));
        }
        for &b in &job_ids {
            actions.push(Kind::Add(b));
            names.push(format!("add:{}", base.buildings[b].id));
            actions.push(Kind::Remove(b));
            names.push(format!("remove:{}", base.buildings[b].id));
        }
        actions.push(Kind::AddBuilder);
        names.push("add_builder:".into());
        actions.push(Kind::RemoveBuilder);
        names.push("remove_builder:".into());
        for (i, t) in base.techs.iter().enumerate() {
            actions.push(Kind::Research(i));
            names.push(format!("research:{}", t.id));
        }
        for (i, l) in base.laws.iter().enumerate() {
            actions.push(Kind::Law(i));
            names.push(format!("law:{}", l.id));
        }
        actions.push(Kind::Scout);
        names.push("scout:".into());
        for (i, o) in CHOICE_OPTIONS.iter().enumerate() {
            actions.push(Kind::Choice(i));
            names.push(format!("choice:{o}"));
        }
        let w = World::new_game(base.clone(), 1, STANDARD);
        let ids = Ids::new(&w);
        let mut e = Env { base, w, ids, actions, names, build_ids, job_ids, actions_this_hour: 0, precursor_obs: false, oracle_obs: false, max_days: MAX_DAYS };
        e.reset(&ResetOpts { seed: 1, no_floor: true, difficulty: "STANDARD".into(), ..Default::default() });
        e
    }

    pub fn n_actions(&self) -> usize {
        self.actions.len()
    }

    // ------------------------------------------------------------ reset

    pub fn reset(&mut self, c: &ResetOpts) -> StepOut {
        let mut defs = self.base.clone();
        let mut rng = Rng::new(c.seed.wrapping_mul(7919).wrapping_add(13) as u64);
        if c.perturb {
            perturb(&mut defs, &mut rng);
        }
        let warm = c.warm;
        if warm != 0.0 {
            for p in defs.profiles.iter_mut() {
                for e in p.script.iter_mut() {
                    e.1 = minf(e.1 + warm, -10.0);
                }
                for st in p.storms.iter_mut() {
                    for t in st.temps.iter_mut() {
                        *t = minf(*t + warm, -10.0);
                    }
                    if let Some(ec) = st.end_c {
                        st.end_c = Some(minf(ec + warm, -10.0));
                    }
                    if let Some((sh, eh, fc)) = st.final_phase {
                        st.final_phase = Some((sh, eh, minf(fc + warm, -10.0)));
                    }
                }
                for t in p.rs_temps.iter_mut() {
                    *t = minf(*t + warm, -10.0);
                }
            }
        }
        if c.no_floor {
            defs.pop_floor = 0.0;
        }
        defs.weather_chaotic = c.weather == "chaotic";
        if !c.weather_noise.is_empty() {
            defs.chaos.set_noise(&c.weather_noise);
        }
        self.precursor_obs = c.precursor_obs;
        self.oracle_obs = c.oracle_obs;
        self.max_days = if c.max_days > 0 { c.max_days } else { MAX_DAYS };
        let diff = DIFFICULTY_KEYS.iter().position(|k| *k == c.difficulty).unwrap_or(STANDARD);
        self.w = World::new_game(defs, c.seed as u64, diff);
        self.ids = Ids::new(&self.w);
        self.actions_this_hour = 0;
        self.result(0.0, false)
    }

    // ------------------------------------------------------------ step

    pub fn step(&mut self, i: i64) -> StepOut {
        let (reward, done) = self.step_raw(i);
        self.result(reward, done)
    }

    /// Advance without building the observation (the caller asks for obs/mask).
    pub fn step_raw(&mut self, i: i64) -> (f64, bool) {
        let mut i = i;
        if i < 0 || i as usize >= self.actions.len() || !self.mask_one(self.actions[i as usize]) {
            i = 0;
        }
        let a = self.actions[i as usize];
        let mut cost = 0.0f64;
        if a == Kind::Bot {
            bot::hourly(&mut self.w, &self.ids);
        } else if a != Kind::Wait {
            self.apply(a);
            cost = ACTION_COST;
            self.actions_this_hour += 1;
            if self.actions_this_hour < MAX_ACTIONS {
                return (-cost, false);
            }
        }
        self.actions_this_hour = 0;
        let dead_before = self.dead();
        self.w.advance(60);
        let after = self.w.alive_count();
        let deaths = self.dead() - dead_before;
        let mut reward = after as f64 / 24.0 - 10.0 * deaths as f64;
        let mut done = after == 0 || self.w.day() >= self.max_days;
        // FP Generator wiki: the explosion ends the game; the living count as lost.
        if self.w.furnace_destroyed || self.w.exiled {
            reward -= 10.0 * after as f64;
            done = true;
        }
        (reward / 10.0 - cost, done)
    }

    fn dead(&self) -> i64 {
        self.w.survivors.iter().filter(|s| !s.alive()).count() as i64
    }

    fn apply(&mut self, a: Kind) {
        match a {
            Kind::Furnace(lv) => self.w.set_furnace_level(lv),
            Kind::Overdrive(on) => {
                self.w.set_overdrive(on);
            }
            Kind::Build(d) => self.build(d),
            Kind::Add(d) => self.add(d),
            Kind::Remove(d) => self.remove(d),
            Kind::AddBuilder => {
                if let Some(si) = self.idle_adult() {
                    let id = self.w.survivors[si].id;
                    self.w.assign_job(id, BUILDER_JOB);
                }
            }
            Kind::RemoveBuilder => {
                let found = self.w.survivors.iter().find(|s| s.alive() && s.job_id == BUILDER_JOB).map(|s| s.id);
                if let Some(id) = found {
                    self.w.assign_job(id, -1);
                }
            }
            Kind::Research(t) => {
                self.w.start_research(t);
            }
            Kind::Law(l) => {
                self.w.sign_law(l);
            }
            Kind::Scout => self.scout(),
            Kind::Choice(o) => {
                self.w.resolve_choice(0, o);
            }
            Kind::Wait | Kind::Bot => {}
        }
    }

    fn build_cell(&self, d: usize) -> Option<(i64, i64)> {
        let ids = &self.ids;
        if d == ids.coal_mine {
            bot::cell_on_seam(&self.w, d)
        } else if d == ids.gathering_post {
            bot::cell_near_piles(&self.w, d, ids)
        } else if d == ids.sawmill {
            bot::cell_near_forest(&self.w, d)
        } else {
            bot::cell_near_furnace(&self.w, d)
        }
    }

    fn can_take(&self, s: &Survivor, bi: usize) -> bool {
        if s.role == CHILD && !self.w.child_may_work_for(s, bi) {
            return false;
        }
        s.role == ENGINEER || !self.w.defs.buildings[self.w.buildings[bi].def].engineers_only
    }

    fn add_ok(&self, d: usize) -> bool {
        match self.target(d) {
            Some(bi) => self.w.survivors.iter().any(|s| s.alive() && s.job_id == -1 && self.can_take(s, bi)),
            None => false,
        }
    }

    fn build_ok(&self, d: usize) -> bool {
        !self.w.building_locked(d) && self.affordable(&self.w.defs.buildings[d].cost) && self.build_cell(d).is_some()
    }

    fn scout_team(&self) -> Vec<i64> {
        let mut ids: Vec<i64> = Vec::new();
        let members = self.w.defs.team_members as i64;
        for s in &self.w.survivors {
            if (ids.len() as i64) < members && s.alive() && s.role == WORKER && s.job_id != SCOUT_JOB && s.illness == NONE {
                ids.push(s.id);
            }
        }
        ids
    }

    fn scout_ok(&self) -> bool {
        let w = &self.w;
        w.scouting_available()
            && self.nearest_site() >= 0
            && self.scout_team().len() as i64 == w.defs.team_members as i64
            && w.defs.team_cost.iter().all(|&(k, c)| !(w.res[k] < c))
    }

    fn build(&mut self, d: usize) {
        if let Some((x, y)) = self.build_cell(d) {
            self.w.order_build(d, x, y);
        }
        if let Some(si) = self.idle_adult() {
            let id = self.w.survivors[si].id;
            self.w.assign_job(id, BUILDER_JOB);
        }
    }

    fn idle_adult(&self) -> Option<usize> {
        self.w.survivors.iter().position(|s| s.alive() && s.job_id == -1 && s.role != CHILD)
    }

    fn crew(&self, bid: i64) -> i64 {
        self.w.survivors.iter().filter(|s| s.alive() && s.job_id == bid).count() as i64
    }

    fn target(&self, d: usize) -> Option<usize> {
        let mut best = None;
        let mut best_free = 0i64;
        let cap = self.w.defs.buildings[d].workers as i64;
        for (bi, b) in self.w.buildings.iter().enumerate() {
            if b.def != d || !b.built {
                continue;
            }
            let free = cap - self.crew(b.id);
            if free > best_free {
                best_free = free;
                best = Some(bi);
            }
        }
        best
    }

    fn add(&mut self, d: usize) {
        let bi = match self.target(d) {
            Some(b) => b,
            None => return,
        };
        let bid = self.w.buildings[bi].id;
        for si in 0..self.w.survivors.len() {
            let s = &self.w.survivors[si];
            if s.alive() && s.job_id == -1 {
                let id = s.id;
                if self.w.assign_job(id, bid) {
                    return;
                }
            }
        }
    }

    fn remove(&mut self, d: usize) {
        for si in 0..self.w.survivors.len() {
            let s = &self.w.survivors[si];
            if s.alive() && s.job_id >= 0 {
                if let Some(bi) = self.w.building_by_id(s.job_id) {
                    if self.w.buildings[bi].def == d {
                        let id = s.id;
                        self.w.assign_job(id, -1);
                        return;
                    }
                }
            }
        }
    }

    fn scout(&mut self) {
        for i in 0..self.w.sites.len() {
            if self.w.sites[i].visited {
                continue;
            }
            if self.w.teams.iter().any(|t| t.site == i as i64) {
                continue;
            }
            let ids = self.scout_team();
            let site = self.nearest_site();
            self.w.send_expedition(site, &ids);
            return;
        }
    }

    fn nearest_site(&self) -> i64 {
        let mut best: i64 = -1;
        for i in 0..self.w.sites.len() {
            let busy = self.w.teams.iter().any(|t| t.site == i as i64);
            if self.w.sites[i].visited || busy {
                continue;
            }
            if best < 0 || self.w.sites[i].hours < self.w.sites[best as usize].hours {
                best = i as i64;
            }
        }
        best
    }

    // ------------------------------------------------------------ mask / obs

    fn affordable(&self, cost: &[(usize, f64)]) -> bool {
        cost.iter().all(|&(k, c)| !(self.w.res[k] < c - 0.0001))
    }

    fn overdrive_ok(&self, on: bool) -> bool {
        let w = &self.w;
        if on {
            !w.overdrive && !w.furnace_destroyed && w.furnace_lit && w.buildings[w.furnace()].level > 0
        } else {
            w.overdrive
        }
    }

    fn mask_one(&self, a: Kind) -> bool {
        let w = &self.w;
        if !w.pending.is_empty() {
            return matches!(a, Kind::Choice(o) if w.pending[0].options.contains(&o));
        }
        match a {
            Kind::Wait | Kind::Bot => true,
            Kind::Furnace(lv) => !w.furnace_destroyed && lv <= w.furnace_max_level() && lv != w.buildings[w.furnace()].level,
            Kind::Overdrive(on) => self.overdrive_ok(on),
            Kind::Build(d) => self.build_ok(d),
            Kind::Add(d) => self.add_ok(d),
            Kind::Remove(d) => w.survivors.iter().any(|s| {
                s.alive() && s.job_id >= 0 && w.building_by_id(s.job_id).map(|bi| w.buildings[bi].def == d).unwrap_or(false)
            }),
            Kind::AddBuilder => self.idle_adult().is_some() && bot::site_pending(w),
            Kind::RemoveBuilder => w.survivors.iter().any(|s| s.alive() && s.job_id == BUILDER_JOB),
            Kind::Research(t) => w.can_research(t),
            Kind::Law(l) => w.laws[l].is_none(),
            Kind::Scout => self.scout_ok(),
            Kind::Choice(_) => false,
        }
    }

    pub fn mask(&self) -> Vec<bool> {
        let mut m = Vec::with_capacity(self.actions.len());
        self.mask_into(&mut m);
        m
    }

    pub fn mask_into(&self, m: &mut Vec<bool>) {
        m.clear();
        let w = &self.w;
        if !w.pending.is_empty() {
            let opts = &w.pending[0].options;
            for &a in &self.actions {
                m.push(matches!(a, Kind::Choice(o) if opts.contains(&o)));
            }
            return;
        }
        let idle = self.idle_adult().is_some();
        let pending_site = bot::site_pending(w);
        let any_builder = w.survivors.iter().any(|s| s.alive() && s.job_id == BUILDER_JOB);
        let fmax = w.furnace_max_level();
        let flevel = w.buildings[w.furnace()].level;
        let nb = w.defs.buildings.len();
        let mut staffed = vec![false; nb];
        for s in &w.survivors {
            if s.alive() && s.job_id >= 0 {
                if let Some(bi) = w.building_by_id(s.job_id) {
                    staffed[w.buildings[bi].def] = true;
                }
            }
        }
        let can_scout = self.scout_ok();
        for &a in &self.actions {
            m.push(match a {
                Kind::Wait | Kind::Bot => true,
                Kind::Furnace(lv) => !w.furnace_destroyed && lv <= fmax && lv != flevel,
                Kind::Overdrive(on) => self.overdrive_ok(on),
                Kind::Build(d) => self.build_ok(d),
                Kind::Add(d) => self.add_ok(d),
                Kind::Remove(d) => staffed[d],
                Kind::AddBuilder => idle && pending_site,
                Kind::RemoveBuilder => any_builder,
                Kind::Research(t) => w.can_research(t),
                Kind::Law(l) => w.laws[l].is_none(),
                Kind::Scout => can_scout,
                Kind::Choice(_) => false,
            });
        }
    }

    pub fn obs(&self) -> Vec<f64> {
        let mut o = Vec::with_capacity(128);
        self.obs_into(&mut o);
        o
    }

    pub fn obs_into(&self, o: &mut Vec<f64>) {
        o.clear();
        let w = &self.w;
        let md = self.max_days as f64; // = MAX_DAYS unless the reset asked for longer episodes
        o.push(w.day() as f64 / md);
        o.push(w.hour() as f64 / 24.0);
        o.push(w.outdoor_temp() / 100.0);
        o.push(w.weather.temp_c / 100.0);
        o.push(if w.weather.in_storm(w.minute) { 1.0 } else { 0.0 });
        o.push(if w.weather.forecast_active(&w.defs, w.minute) { 1.0 } else { 0.0 });
        o.push(clampf((w.weather.next_storm_start - w.minute) as f64 / (1440.0 * 30.0), -1.0, 1.0));
        o.push(w.weather.coldest_storm_temp() / 100.0);
        let f = w.furnace();
        o.push(w.buildings[f].level as f64 / 4.0);
        o.push(if w.furnace_lit { 1.0 } else { 0.0 });
        o.push(w.furnace_max_level() as f64 / 4.0);
        o.push(w.heater_max_level() as f64 / 3.0);
        for k in RES_OBS {
            o.push((1.0 + w.res[k]).ln() / 6.0);
        }
        let mut roles = [0i64; 3];
        let (mut sick, mut grave, mut hungry, mut homeless, mut idle, mut builders, mut alive) = (0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
        let mut hypo = 0.0f64;
        for s in &w.survivors {
            if !s.alive() {
                continue;
            }
            alive += 1;
            roles[s.role as usize] += 1;
            if s.illness == SICK {
                sick += 1;
            } else if s.illness != NONE {
                grave += 1;
            }
            hypo += s.hypo;
            if s.unfed_days > 0.0 {
                hungry += 1;
            }
            if s.bed_id < 0 {
                homeless += 1;
            }
            if s.job_id == -1 && s.role != CHILD {
                idle += 1;
            }
            if s.job_id == BUILDER_JOB {
                builders += 1;
            }
        }
        let n = maxf(1.0, alive as f64);
        o.push(alive as f64 / 50.0);
        for r in roles {
            o.push(r as f64 / 50.0);
        }
        for v in [sick as f64 / n, grave as f64 / n, hypo / n, hungry as f64 / n, homeless as f64 / n, idle as f64 / n, builders as f64 / n] {
            o.push(v);
        }
        for &d in &self.job_ids {
            let (mut built, mut crew, mut cap) = (0i64, 0i64, 0i64);
            for b in &w.buildings {
                if b.def == d && b.built {
                    built += 1;
                    crew += self.crew(b.id);
                    cap += w.defs.buildings[d].workers as i64;
                }
            }
            o.push(built as f64 / 5.0);
            o.push(crew as f64 / maxf(1.0, cap as f64));
        }
        for &d in &self.build_ids {
            let (mut c, mut pending) = (0i64, 0i64);
            for b in &w.buildings {
                if b.def == d {
                    if b.built {
                        c += 1;
                    } else {
                        pending += 1;
                    }
                }
            }
            o.push(c as f64 / 10.0);
            o.push(pending as f64);
        }
        for t in 0..w.defs.techs.len() {
            o.push(if w.has_tech(t) { 1.0 } else if w.research_id == Some(t) { 0.5 } else { 0.0 });
        }
        for l in 0..w.defs.laws.len() {
            o.push(if w.laws[l].is_some() { 1.0 } else { 0.0 });
        }
        o.push(w.teams.len() as f64);
        o.push(w.incoming.len() as f64);
        let visited = w.sites.iter().filter(|s| s.visited).count() as i64;
        o.push(visited as f64 / (1i64.max(w.sites.len() as i64)) as f64);
        o.push(if !w.pending.is_empty() { 1.0 } else { 0.0 });
        o.push(self.actions_this_hour as f64 / MAX_ACTIONS as f64);
        o.push(w.hope / 100.0);
        o.push(w.discontent / 100.0);
        o.push(if w.overdrive { 1.0 } else { 0.0 });
        o.push(w.pressure / 100.0);
        o.push(if w.scenario_won { 1.0 } else { 0.0 });
        if self.precursor_obs {
            let wx = &w.weather;
            let ch = wx.chaotic;
            o.push(if ch { (wx.pressure - 10130) as f64 / 300.0 } else { 0.0 });
            o.push(if ch { wx.pressure_trend() as f64 / 300.0 } else { 0.0 });
            o.push(if ch { wx.wind_dir as f64 / 7.0 } else { 0.0 });
            o.push(if ch { wx.wind_ms as f64 / 25.0 } else { 0.0 });
            o.push(if ch { wx.animals as f64 / 2.0 } else { 0.0 });
            o.push(if ch { wx.sky as f64 / 3.0 } else { 0.0 });
        }
        if self.oracle_obs {
            let wx = &w.weather;
            let rs = wx.chaotic;
            o.push(if rs { clampf((wx.oracle_land - w.minute) as f64 / 60.0 / 240.0, 0.0, 1.0) } else { 0.0 });
            o.push(if rs { wx.oracle_tier as f64 } else { 0.0 });
        }
    }

    pub fn result(&self, reward: f64, done: bool) -> StepOut {
        StepOut { obs: self.obs(), mask: self.mask(), reward, done, day: self.w.day(), alive: self.w.alive_count() }
    }

    // ------------------------------------------------------------ parity helpers

    /// The exact reply line rl_server.gd writes (JSON.stringify with sorted keys).
    pub fn line(out: &StepOut) -> String {
        let mut s = String::with_capacity(4096);
        s.push_str("{\"done\":");
        s.push_str(if out.done { "true" } else { "false" });
        s.push_str(&format!(",\"info\":{{\"alive\":{},\"day\":{}}},\"mask\":[", out.alive, out.day));
        for (i, m) in out.mask.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(if *m { "true" } else { "false" });
        }
        s.push_str("],\"obs\":[");
        for (i, x) in out.obs.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&godot_json_num(*x));
        }
        s.push_str("],\"reward\":");
        s.push_str(&godot_json_num(out.reward));
        s.push('}');
        s
    }

    /// Full-state digest with exactly the layout of tools/rl_trace.gd `_digest`.
    pub fn digest(&self) -> (Vec<i64>, Vec<f64>) {
        let w = &self.w;
        let mut ints: Vec<i64> = Vec::new();
        let mut fl: Vec<f64> = Vec::new();
        ints.extend_from_slice(&[
            w.minute,
            w.rng.state as i64,
            w.next_id,
            w.furnace_lit as i64,
            w.heaters_fuelled as i64,
            w.difficulty as i64,
            self.actions_this_hour,
            w.research_id.map(|t| t as i64).unwrap_or(-1),
            w.weather.storm_index,
            w.weather.next_storm_start,
            w.weather.next_storm_end,
            w.weather.storms_survived,
            w.weather.next_storm_temps.len() as i64,
            w.weather.next_storm_final_start,
            w.weather.next_storm_scenario_end as i64,
            w.overdrive as i64,
            w.pressure_events,
            w.furnace_destroyed as i64,
            w.scenario_won as i64,
        ]);
        let wx = &w.weather;
        ints.extend_from_slice(&[
            wx.chaotic as i64,
            if wx.chaotic { wx.wrng.state as i64 } else { 0 },
            wx.lx,
            wx.ly,
            wx.lz,
            (wx.lx < 0) as i64, // 0, -1, -1, 0 and this flag: slots of the retired anchored weather
            wx.phase,
            0,
            -1,
            -1,
            0,
            wx.tier,
            wx.pressure,
            wx.wind_dir,
            wx.wind_ms,
            wx.animals,
            wx.sky,
            wx.last_onset,
            wx.p_hist.len() as i64,
        ]);
        ints.extend_from_slice(&wx.p_hist);
        if wx.chaotic {
            ints.extend_from_slice(&[wx.zh1, wx.zh2, wx.last_end, wx.peak, wx.oracle_land, wx.oracle_tier, wx.z_hist.len() as i64]);
            ints.extend_from_slice(&wx.z_hist);
        }
        ints.push(w.buildings.len() as i64);
        for b in &w.buildings {
            ints.extend_from_slice(&[b.id, b.def as i64, b.cx, b.cy, b.built as i64, b.level, b.heater]);
        }
        ints.push(w.survivors.len() as i64);
        for s in &w.survivors {
            ints.extend_from_slice(&[
                s.id,
                s.age,
                s.role as i64,
                s.trait_ as i64,
                s.act as i64,
                s.next_act as i64,
                s.place_id,
                s.dest_place_id,
                s.job_id,
                s.bed_id,
                s.illness as i64,
                s.frostbites,
                s.died_day,
                s.gear[0],
                s.gear[1],
                s.gear[2],
                s.break_until,
                s.calm_until,
                s.partner_id,
                s.pregnant_since,
                s.parents.iter().filter(|&&p| p >= 0).count() as i64,
            ]);
        }
        for l in &w.laws {
            ints.push(l.unwrap_or(-1));
        }
        ints.push(w.extended.len() as i64);
        ints.extend_from_slice(&w.extended);
        for st in &w.sites {
            ints.push(st.visited as i64);
        }
        ints.push(w.teams.len() as i64);
        for t in &w.teams {
            ints.extend_from_slice(&[t.site, t.back as i64, t.eta, t.members.len() as i64]);
            ints.extend_from_slice(&t.members);
        }
        ints.push(w.incoming.len() as i64);
        for g in &w.incoming {
            ints.extend_from_slice(&[g.arrive, g.roles.len() as i64]);
            for &r in &g.roles {
                ints.push(r as i64);
            }
        }
        ints.push(w.researched.len() as i64);
        for &t in &w.researched {
            ints.push(t as i64);
        }
        ints.push(w.pending.len() as i64);
        for c in &w.pending {
            ints.extend_from_slice(&[c.kind as i64, c.count, c.sick, c.healthy, c.day, c.options.len() as i64]);
        }
        ints.extend_from_slice(&[w.ultimatum, w.ultimatum_end, w.exiled as i64, w.chapter]);
        ints.push(w.gear_stock.len() as i64);
        for &(k, q) in &w.gear_stock {
            ints.extend_from_slice(&[k, q]);
        }
        for &t in &w.tiles {
            ints.push(t as i64);
        }
        for k in RES_OBS {
            fl.push(w.res[k]);
        }
        fl.extend_from_slice(&[
            w.hope,
            w.discontent,
            w.research_done_h,
            w.weather.temp_c,
            if w.weather.next_storm_end_c.is_nan() { -12345.0 } else { w.weather.next_storm_end_c },
            if w.weather.next_storm_final_c.is_nan() { -12345.0 } else { w.weather.next_storm_final_c },
            w.pressure,
        ]);
        fl.extend_from_slice(&w.weather.next_storm_temps);
        for b in &w.buildings {
            fl.push(b.progress);
            fl.push(b.amount);
        }
        for s in &w.survivors {
            fl.extend_from_slice(&[
                s.pos.x as f64,
                s.pos.y as f64,
                s.target.x as f64,
                s.target.y as f64,
                s.body_level,
                s.sick_chance,
                s.illness_days,
                s.cure_progress,
                s.hypo,
                s.unfed_days,
                s.age_days,
            ]);
        }
        for st in &w.sites {
            fl.push(st.hours);
        }
        fl.extend_from_slice(&w.tile_amount);
        (ints, fl)
    }
}

/// Training-only domain randomization, exactly as rl_server.gd `_perturb`.
fn perturb(defs: &mut Defs, rng: &mut Rng) {
    for p in defs.profiles.iter_mut() {
        let mut script: Vec<(f64, f64)> = Vec::new();
        for e in &p.script {
            let mut day = e.0 as i64;
            if day > 0 {
                day = 1i64.max(day + rng.randi_range(-2, 2));
            }
            let t = e.1 + 10.0 * rng.randi_range(-1, 1) as f64;
            script.push((day as f64, t));
        }
        // Godot's sort is insertion sort (stable) for <= 16 elements.
        assert!(script.len() <= 16, "script longer than Godot's insertion-sort threshold: port introsort");
        script.sort_by(|a, b| (a.0 as i64).cmp(&(b.0 as i64)));
        p.script = script;
        for st in p.storms.iter_mut() {
            st.day = (st.day as i64 + rng.randi_range(-3, 3)) as f64;
            let mut temps = st.temps.clone();
            let n = rng.randi_range(-1, 1);
            if n < 0 && temps.len() > 2 {
                temps.remove(1);
            } else if n > 0 {
                let t1 = temps[1];
                temps.insert(1, t1);
            }
            st.temps = temps;
        }
    }
    for d in defs.buildings.iter_mut() {
        if let Some((r, amount)) = d.pile {
            d.pile = Some((r, amount * rng.randf_range(0.85, 1.15)));
        }
    }
    for sd in defs.sites.iter_mut() {
        let a = sd.hours[0] * rng.randf_range(0.85, 1.15);
        let b = sd.hours[1] * rng.randf_range(0.85, 1.15);
        sd.hours = [a, b];
    }
}

/// Godot JSON.stringify for a float (full_precision = false).
pub fn godot_json_num(x: f64) -> String {
    if !x.is_finite() {
        return if x == f64::INFINITY {
            "1e99999".into()
        } else if x == f64::NEG_INFINITY {
            "-1e99999".into()
        } else {
            "null".into()
        };
    }
    if x == 0.0 {
        return "0.0".into();
    }
    let magnitude = x.abs().log10();
    let precision = 1i32.max(14 - magnitude.floor() as i32);
    let mut s = c_fixed(x, precision.min(32));
    if s.contains('.') {
        let bytes = unsafe { s.as_bytes_mut() };
        let mut z = bytes.len() - 1;
        let mut cut = bytes.len();
        while z > 0 {
            if bytes[z] == b'0' {
                cut = z;
            } else if bytes[z] == b'.' {
                cut = z + 2;
                break;
            } else {
                break;
            }
            z -= 1;
        }
        s.truncate(cut);
    }
    s
}

/// printf("%.<p>lf") from the C library, as Godot's String::num uses.
fn c_fixed(x: f64, p: i32) -> String {
    let mut buf = [0u8; 400];
    let fmt = b"%.*f\0";
    let n = unsafe { libc::snprintf(buf.as_mut_ptr() as *mut libc::c_char, buf.len(), fmt.as_ptr() as *const libc::c_char, p as libc::c_int, x) };
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

/// The f64 Python gets after json.loads of Godot's text.
pub fn json_f64(x: f64) -> f64 {
    godot_json_num(x).parse::<f64>().unwrap_or(f64::NAN)
}

/// The f32 Python gets after np.array(json.loads(...), np.float32), computed
/// without formatting unless the decimal rounding could move the f32 result.
#[inline]
pub fn obs_f32(x: f64) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let f = x as f32;
    if !f.is_finite() || f == 0.0 {
        return json_f64(x) as f32;
    }
    let fd = f as f64;
    let up = (f32_next_up(f) as f64 - fd) * 0.5;
    let down = (fd - f32_next_down(f) as f64) * 0.5;
    let tol = x.abs() * 1e-13;
    let d_hi = (fd + up - x).abs();
    let d_lo = (x - (fd - down)).abs();
    if d_hi > tol && d_lo > tol {
        f
    } else {
        json_f64(x) as f32
    }
}

fn f32_next_up(f: f32) -> f32 {
    let b = f.to_bits();
    if f > 0.0 {
        f32::from_bits(b + 1)
    } else {
        f32::from_bits(b - 1)
    }
}
fn f32_next_down(f: f32) -> f32 {
    let b = f.to_bits();
    if f > 0.0 {
        f32::from_bits(b - 1)
    } else {
        f32::from_bits(b + 1)
    }
}
