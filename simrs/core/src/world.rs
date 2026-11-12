//! Port of sim/sim_world.gd, sim/survivor.gd and sim/building.gd (rules only;
//! no save/load, no event text). Every expression keeps GDScript's operation
//! order and types: `float` = f64, Vector2 = f32, `int` = i64.

use crate::defs::{Defs, Shift, NRES};
use crate::rng::Rng;
use crate::weather::{clampf, maxf, minf, Weather};

pub const BUILDER_JOB: i64 = -2;
pub const SCOUT_JOB: i64 = -3;

pub const SNOW: u8 = 0;
pub const FOREST: u8 = 1;
pub const COAL_SEAM: u8 = 2;

pub const GENTLE: usize = 0;
pub const STANDARD: usize = 1;
pub const HARSH: usize = 2;
pub const HARDCORE: usize = 3;

pub const N_TRAITS: i64 = 5; // ["hardy", "frail", "diligent", "brave", "kind"]
pub const TRAIT_FRAIL: u8 = 1;

pub const TAU: f64 = std::f64::consts::TAU;
pub const N_NAMES: i64 = 30;
pub const FEMALE: [u8; 30] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1];

/// GDScript SimWorld.curve: piecewise-linear, clamped at both ends.
pub fn curve(points: &[(f64, f64)], x: f64) -> f64 {
    if x <= points[0].0 {
        return points[0].1;
    }
    for i in 1..points.len() {
        let x1 = points[i].0;
        if x <= x1 {
            let x0 = points[i - 1].0;
            let y0 = points[i - 1].1;
            return y0 + (points[i].1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    points[points.len() - 1].1
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Act {
    Idle = 0,
    Walk,
    Work,
    Warm,
    Sleep,
    Rest,
    Collapsed,
    Away,
    Dead,
}

pub const WORKER: u8 = 0;
pub const ENGINEER: u8 = 1;
pub const CHILD: u8 = 2;

pub const NONE: u8 = 0;
pub const SICK: u8 = 1;
pub const GRAVE: u8 = 2;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct V2 {
    pub x: f32,
    pub y: f32,
}

impl V2 {
    #[inline]
    pub fn new(x: f32, y: f32) -> V2 {
        V2 { x, y }
    }
    /// Vector2(a, b) built from GDScript floats.
    #[inline]
    pub fn f(x: f64, y: f64) -> V2 {
        V2 { x: x as f32, y: y as f32 }
    }
    #[inline]
    pub fn add(self, o: V2) -> V2 {
        V2::new(self.x + o.x, self.y + o.y)
    }
    #[inline]
    pub fn sub(self, o: V2) -> V2 {
        V2::new(self.x - o.x, self.y - o.y)
    }
    /// Vector2 * float: the float is narrowed to real_t first.
    #[inline]
    pub fn mul(self, s: f64) -> V2 {
        let s = s as f32;
        V2::new(self.x * s, self.y * s)
    }
    #[inline]
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    #[inline]
    pub fn distance_to(self, o: V2) -> f32 {
        ((self.x - o.x) * (self.x - o.x) + (self.y - o.y) * (self.y - o.y)).sqrt()
    }
    #[inline]
    pub fn normalized(self) -> V2 {
        if !(self.x.is_finite() && self.y.is_finite()) {
            return V2::default();
        }
        let l = self.x * self.x + self.y * self.y;
        if l == 0.0 {
            V2::default()
        } else {
            let l = l.sqrt();
            V2::new(self.x / l, self.y / l)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Building {
    pub id: i64,
    pub def: usize,
    pub cx: i64,
    pub cy: i64,
    pub built: bool,
    pub progress: f64,
    pub level: i64,
    pub amount: f64,
    pub heater: i64,
}

#[derive(Clone, Debug)]
pub struct Survivor {
    pub id: i64,
    pub age: i64,
    pub role: u8,
    pub trait_: u8,
    pub pos: V2,
    pub target: V2,
    pub act: Act,
    pub next_act: Act,
    pub place_id: i64,
    pub dest_place_id: i64,
    pub job_id: i64,
    pub bed_id: i64,
    pub body_level: f64,
    pub sick_chance: f64,
    pub illness: u8,
    pub illness_days: f64,
    pub cure_progress: f64,
    pub hypo: f64,
    pub frostbites: i64,
    pub unfed_days: f64,
    pub died_day: i64,
    pub gear: [i64; 3], // quality per kind (parka, tuque, tool), -1 = none
    pub break_until: i64,
    pub calm_until: i64,
    pub age_days: f64,
    pub pregnant_since: i64,
    pub partner_id: i64,
    pub parents: [i64; 2],
}

impl Survivor {
    #[inline]
    pub fn alive(&self) -> bool {
        self.act != Act::Dead
    }
    #[inline]
    pub fn lost_digits(&self) -> i64 {
        self.frostbites / 2
    }
}

#[derive(Clone, Debug)]
pub struct Site {
    pub def: usize,
    pub hours: f64,
    pub visited: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Team {
    pub members: Vec<i64>,
    pub site: i64,
    pub back: bool,
    pub eta: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub arrive: i64,
    pub roles: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Choice {
    pub kind: u8, // 0 refugees, 1 generator pressure
    pub options: Vec<usize>, // indices into CHOICE_OPTIONS
    pub count: i64,
    pub sick: i64,
    pub healthy: i64,
    pub day: i64,
}

pub const CHOICE_OPTIONS: [&str; 5] = ["accept", "accept_healthy", "refuse", "sacrifice_core", "sacrifice_child"];
pub const OPT_CORE: usize = 3;
pub const OPT_CHILD: usize = 4;

#[derive(Clone)]
pub struct World {
    pub defs: Defs,
    pub rng: Rng,
    pub minute: i64,
    pub map_size: i64,
    pub tiles: Vec<u8>,
    pub tile_amount: Vec<f64>,
    pub buildings: Vec<Building>,
    pub survivors: Vec<Survivor>,
    pub res: [f64; NRES],
    pub weather: Weather,
    pub hope: f64,
    pub discontent: f64,
    pub needs_discontent: f64,
    pub difficulty: usize,
    pub offline: bool,
    pub storm_shelter: bool,
    pub pending: Vec<Choice>,
    pub next_id: i64,
    pub furnace_lit: bool,
    pub laws: Vec<Option<i64>>, // law index -> day signed
    pub extended: Vec<i64>,
    pub sites: Vec<Site>,
    pub teams: Vec<Team>,
    pub incoming: Vec<Group>,
    pub researched: Vec<usize>,
    pub research_id: Option<usize>,
    pub research_done_h: f64,
    pub heaters_fuelled: bool,
    pub overdrive: bool,
    pub pressure: f64,
    pub pressure_events: i64,
    pub furnace_destroyed: bool,
    pub scenario_won: bool,
    pub gear_stock: Vec<(i64, i64)>, // unworn gear: (kind, quality)
    pub ultimatum: i64, // 0 none, 1 discontent, 2 hope
    pub ultimatum_end: i64,
    pub exiled: bool,
    pub chapter: i64,
    occ: Vec<u16>,
    bidx: Vec<i32>,
    sidx: Vec<i32>,
    furnace_def: usize,
}

#[inline]
fn fi(x: f64) -> i64 {
    // GDScript int(float): truncation toward zero
    x as i64
}

impl World {
    // ============================================================ setup

    pub fn new_game(defs: Defs, seed: u64, difficulty: usize) -> World {
        let map_size = fi(defs.map_size);
        let nlaws = defs.laws.len();
        let furnace_def = defs.bix("furnace");
        let mut w = World {
            rng: Rng::new(seed),
            minute: 0,
            map_size,
            tiles: Vec::new(),
            tile_amount: Vec::new(),
            buildings: Vec::new(),
            survivors: Vec::new(),
            res: [0.0; NRES],
            weather: Weather::new(),
            hope: defs.start_hope,
            discontent: defs.start_discontent,
            needs_discontent: 0.0,
            difficulty,
            offline: false,
            storm_shelter: true,
            pending: Vec::new(),
            next_id: 1,
            furnace_lit: false,
            laws: vec![None; nlaws],
            extended: Vec::new(),
            sites: Vec::new(),
            teams: Vec::new(),
            incoming: Vec::new(),
            researched: Vec::new(),
            research_id: None,
            research_done_h: 0.0,
            heaters_fuelled: true,
            overdrive: false,
            pressure: 0.0,
            pressure_events: 0,
            furnace_destroyed: false,
            scenario_won: false,
            gear_stock: Vec::new(),
            ultimatum: 0,
            ultimatum_end: -1,
            exiled: false,
            chapter: 0,
            occ: vec![0; (map_size * map_size) as usize],
            bidx: Vec::new(),
            sidx: Vec::new(),
            furnace_def,
            defs,
        };
        for &(k, v) in &w.defs.start_resources {
            w.res[k] = v;
        }
        let prof = w.defs.profile_by_difficulty[difficulty];
        w.weather.init(&w.defs, prof, seed);
        w.generate_map();
        for i in 0..w.defs.sites.len() {
            let h = w.defs.sites[i].hours;
            let hours = w.rng.randf_range(h[0], h[1]);
            w.sites.push(Site { def: i, hours, visited: false });
        }
        let mut i = 0i64;
        for role in [WORKER, ENGINEER, CHILD] {
            for _ in 0..fi(w.defs.start_roles[role as usize]) {
                let at = w.fire_spot(i);
                w.add_survivor(at, role);
                i += 1;
            }
        }
        w
    }

    fn generate_map(&mut self) {
        let n = (self.map_size * self.map_size) as usize;
        self.tiles = vec![SNOW; n];
        self.tile_amount = vec![0.0; n];
        let c = self.map_center();
        let fur = self.place(self.furnace_def, c.x as f64 as i64 - 1, c.y as f64 as i64 - 1, true);
        assert!(fur.is_some());
        let ms = self.map_size;
        for _ in 0..fi(self.defs.forest_patches) {
            let p = self.ring_point();
            let (py, px) = (p.y as f64 as i64, p.x as f64 as i64);
            for y in (py - 4)..(py + 5) {
                for x in (px - 4)..(px + 5) {
                    if self.in_bounds(x, y)
                        && !self.occupied(x, y)
                        && (V2::f(x as f64 + 0.5, y as f64 + 0.5).distance_to(p) as f64) <= 2.6
                    {
                        self.tiles[(y * ms + x) as usize] = FOREST;
                        self.tile_amount[(y * ms + x) as usize] = self.defs.forest_tile_amount;
                    }
                }
            }
        }
        for _ in 0..fi(self.defs.coal_seams) {
            for _attempt in 0..50 {
                let p = self.ring_point();
                let (cx, cy) = (p.x as f64 as i64, p.y as f64 as i64);
                let mut ok = true;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (x, y) = (cx + dx, cy + dy);
                        if !self.in_bounds(x, y) || self.occupied(x, y) || self.tile_at(x, y) != SNOW {
                            ok = false;
                        }
                    }
                }
                if ok {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            self.tiles[((cy + dy) * ms + cx + dx) as usize] = COAL_SEAM;
                        }
                    }
                    break;
                }
            }
        }
        let specs = [
            (self.defs.bix("coal_pile"), self.defs.coal_piles),
            (self.defs.bix("wood_crates"), self.defs.wood_crates),
            (self.defs.bix("scrap_heap"), self.defs.scrap_heaps),
        ];
        for (d, count) in specs {
            for _ in 0..fi(count) {
                for _attempt in 0..50 {
                    let p = self.ring_point();
                    let (cx, cy) = (p.x as f64 as i64, p.y as f64 as i64);
                    if self.can_place(d, cx, cy) && self.tile_at(cx, cy) == SNOW {
                        let bi = self.place(d, cx, cy, true).unwrap();
                        self.buildings[bi].amount = self.defs.buildings[d].pile.unwrap().1;
                        break;
                    }
                }
            }
        }
    }

    fn ring_point(&mut self) -> V2 {
        let ang = self.rng.randf() * TAU;
        let r = self.rng.randf_range(self.defs.ring_min, self.defs.ring_max);
        self.map_center().add(V2::f(ang.cos(), ang.sin()).mul(r))
    }

    pub fn add_survivor(&mut self, at: V2, role: u8) -> usize {
        let id = self.take_id();
        let age = if role == CHILD { self.rng.randi_range(6, 13) } else { self.rng.randi_range(16, 62) };
        let tr = self.rng.randi_range(0, N_TRAITS - 1) as u8;
        let body = clampf(self.ground_level(at), -3.0, 2.0);
        self.survivors.push(Survivor {
            id,
            age,
            role,
            trait_: tr,
            pos: at,
            target: at,
            act: Act::Idle,
            next_act: Act::Idle,
            place_id: -1,
            dest_place_id: -1,
            job_id: -1,
            bed_id: -1,
            body_level: body,
            sick_chance: 0.0,
            illness: NONE,
            illness_days: 0.0,
            cure_progress: 0.0,
            hypo: 0.0,
            frostbites: 0,
            unfed_days: 0.0,
            died_day: -1,
            gear: [-1; 3],
            break_until: -1,
            calm_until: -1,
            age_days: ((id * 13) % 60) as f64,
            pregnant_since: -1,
            partner_id: -1,
            parents: [-1; 2],
        });
        let si = self.survivors.len() - 1;
        if self.sidx.len() <= id as usize {
            self.sidx.resize(id as usize + 1, -1);
        }
        self.sidx[id as usize] = si as i32;
        si
    }

    fn take_id(&mut self) -> i64 {
        self.next_id += 1;
        self.next_id - 1
    }

    // ============================================================ map & buildings

    #[inline]
    pub fn map_center(&self) -> V2 {
        V2::f(self.map_size as f64 * 0.5, self.map_size as f64 * 0.5)
    }

    #[inline]
    pub fn in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && x < self.map_size && y < self.map_size
    }

    #[inline]
    pub fn tile_at(&self, x: i64, y: i64) -> u8 {
        if self.in_bounds(x, y) {
            self.tiles[(y * self.map_size + x) as usize]
        } else {
            SNOW
        }
    }

    /// Only called for in-bounds cells (as in GDScript); kept as an occupancy grid.
    #[inline]
    pub fn occupied(&self, x: i64, y: i64) -> bool {
        self.occ[(y * self.map_size + x) as usize] > 0
    }

    fn mark(&mut self, bi: usize, delta: i32) {
        let b = &self.buildings[bi];
        let size = self.defs.buildings[b.def].size;
        // GDScript: c.x >= cell.x and c.x < cell.x + size[0] (float compare)
        let (sx, sy) = (size[0].ceil() as i64, size[1].ceil() as i64);
        let (cx, cy) = (b.cx, b.cy);
        for y in cy..cy + sy {
            for x in cx..cx + sx {
                if self.in_bounds(x, y) {
                    let i = (y * self.map_size + x) as usize;
                    self.occ[i] = (self.occ[i] as i32 + delta) as u16;
                }
            }
        }
    }

    pub fn can_place(&self, d: usize, cx: i64, cy: i64) -> bool {
        let def = &self.defs.buildings[d];
        for y in 0..fi(def.size[1]) {
            for x in 0..fi(def.size[0]) {
                let (px, py) = (cx + x, cy + y);
                if !self.in_bounds(px, py) || self.occupied(px, py) || self.tile_at(px, py) == FOREST {
                    return false;
                }
                if def.needs_coal_seam != (self.tile_at(px, py) == COAL_SEAM) {
                    return false;
                }
            }
        }
        true
    }

    pub fn place(&mut self, d: usize, cx: i64, cy: i64, built: bool) -> Option<usize> {
        if !self.can_place(d, cx, cy) {
            return None;
        }
        let id = self.take_id();
        self.buildings.push(Building { id, def: d, cx, cy, built, progress: 0.0, level: 0, amount: 0.0, heater: 0 });
        let bi = self.buildings.len() - 1;
        self.mark(bi, 1);
        self.reindex();
        Some(bi)
    }

    pub fn order_build(&mut self, d: usize, cx: i64, cy: i64) -> Option<usize> {
        if self.building_locked(d) {
            return None;
        }
        let cost = self.defs.buildings[d].cost.clone();
        for &(k, c) in &cost {
            if self.res[k] < c - 0.0001 {
                return None;
            }
        }
        if !self.can_place(d, cx, cy) {
            return None;
        }
        for &(k, c) in &cost {
            self.res[k] -= c;
        }
        self.place(d, cx, cy, false)
    }

    fn reindex(&mut self) {
        let n = self.next_id as usize + 1;
        if self.bidx.len() < n {
            self.bidx.resize(n, -1);
        }
        for v in self.bidx.iter_mut() {
            *v = -1;
        }
        for (i, b) in self.buildings.iter().enumerate() {
            self.bidx[b.id as usize] = i as i32;
        }
    }

    #[inline]
    pub fn building_by_id(&self, bid: i64) -> Option<usize> {
        if bid < 0 || bid as usize >= self.bidx.len() {
            return None;
        }
        let i = self.bidx[bid as usize];
        if i < 0 {
            None
        } else {
            Some(i as usize)
        }
    }

    pub fn first_building(&self, d: usize) -> Option<usize> {
        self.buildings.iter().position(|b| b.def == d)
    }

    #[inline]
    pub fn furnace(&self) -> usize {
        if !self.buildings.is_empty() && self.buildings[0].def == self.furnace_def {
            0
        } else {
            self.first_building(self.furnace_def).unwrap()
        }
    }

    #[inline]
    pub fn center(&self, bi: usize) -> V2 {
        let b = &self.buildings[bi];
        let size = self.defs.buildings[b.def].size;
        V2::f(b.cx as f64 + size[0] * 0.5, b.cy as f64 + size[1] * 0.5)
    }

    pub fn fire_spot(&self, i: i64) -> V2 {
        let ang = i as f64 * 2.39996;
        self.center(self.furnace()).add(V2::f(ang.cos(), ang.sin()).mul(1.6 + 0.15 * (i % 4) as f64))
    }

    pub fn work_spot(&self, bi: usize, sid: i64) -> V2 {
        let size = self.defs.buildings[self.buildings[bi].def].size;
        let ang = sid as f64 * 2.39996;
        let r = maxf(size[0], size[1]) * 0.5 + 0.45;
        self.center(bi).add(V2::f(ang.cos(), ang.sin()).mul(r))
    }

    pub fn set_furnace_level(&mut self, level: i64) {
        if self.furnace_destroyed {
            return;
        }
        let f = self.furnace();
        self.buildings[f].level = level.clamp(0, self.furnace_max_level());
        self.update_furnace();
    }

    pub fn set_overdrive(&mut self, on: bool) -> bool {
        if on && (self.furnace_destroyed || self.pressure_choice_pending()) {
            return false;
        }
        self.overdrive = on;
        true
    }

    pub fn pressure_choice_pending(&self) -> bool {
        self.pending.iter().any(|c| c.kind == 1)
    }

    #[inline]
    pub fn overdrive_active(&self) -> bool {
        self.overdrive && self.furnace_lit && self.buildings[self.furnace()].level > 0
    }

    pub fn furnace_heat(&self) -> f64 {
        let mut h = self.buildings[self.furnace()].level as f64;
        if self.overdrive_active() {
            let mut m = self.defs.od_levels;
            for &t in &self.researched {
                if let Some(v) = self.defs.techs[t].overdrive_levels {
                    m = maxf(m, v);
                }
            }
            h += m;
        }
        h
    }

    fn update_pressure(&mut self) {
        if self.overdrive_active() {
            let mut p = 1.0;
            for &t in &self.researched {
                if let Some(v) = self.defs.techs[t].pressure_up_mult {
                    p *= v;
                }
            }
            self.pressure += self.defs.od_up_h * p / 60.0;
        } else {
            let mut p = 1.0;
            for &t in &self.researched {
                if let Some(v) = self.defs.techs[t].pressure_down_mult {
                    p *= v;
                }
            }
            self.pressure = maxf(0.0, self.pressure - self.defs.od_down_h * p / 60.0);
        }
        if self.pressure < self.defs.od_max_pressure {
            return;
        }
        self.pressure = self.defs.od_max_pressure;
        self.overdrive = false;
        self.pressure_events += 1;
        let mut options = Vec::new();
        if self.res[5] >= 1.0 {
            options.push(OPT_CORE);
        }
        if self.sacrifice_child().is_some() {
            options.push(OPT_CHILD);
        }
        if self.pressure_events >= 2 || options.is_empty() {
            self.explode();
            return;
        }
        let day = self.day();
        self.pending.push(Choice { kind: 1, options, count: 0, sick: 0, healthy: 0, day });
    }

    fn sacrifice_child(&self) -> Option<i64> {
        self.survivors.iter().find(|s| s.alive() && s.role == CHILD && s.job_id != SCOUT_JOB).map(|s| s.id)
    }

    fn explode(&mut self) {
        self.furnace_destroyed = true;
        self.overdrive = false;
        let f = self.furnace();
        self.buildings[f].level = 0;
        self.furnace_lit = false;
    }

    pub fn assign_job(&mut self, sid: i64, job: i64) -> bool {
        let si = match self.survivor_by_id(sid) {
            Some(i) => i,
            None => return false,
        };
        if !self.survivors[si].alive() {
            return false;
        }
        let role = self.survivors[si].role;
        if role == CHILD && job < 0 {
            if job == -1 {
                self.survivors[si].job_id = -1;
                return true;
            }
            return false;
        }
        if job >= 0 {
            let bi = match self.building_by_id(job) {
                Some(b) => b,
                None => return false,
            };
            if role == CHILD && !self.child_may_work_for(&self.survivors[si], bi) {
                return false;
            }
            if self.defs.buildings[self.buildings[bi].def].engineers_only && role != ENGINEER {
                return false;
            }
        }
        self.survivors[si].job_id = job;
        true
    }

    pub fn has_law(&self, l: Option<usize>) -> bool {
        match l {
            Some(i) => self.laws[i].is_some(),
            None => false,
        }
    }

    pub fn sign_law(&mut self, l: usize) -> bool {
        if self.laws[l].is_some() {
            return false;
        }
        let pts = self.defs.laws_points_per_tier;
        self.laws[l] = Some(self.day());
        let (h, d) = (self.defs.laws[l].hope_tiers, self.defs.laws[l].discontent_tiers);
        self.hope = clampf(self.hope + h * pts, 0.0, 100.0);
        self.discontent = clampf(self.discontent + d * pts, 0.0, 100.0);
        true
    }

    pub fn set_extended_shift(&mut self, bid: i64, on: bool) -> bool {
        if !self.has_law(self.defs.law_extended) {
            return false;
        }
        let bi = match self.building_by_id(bid) {
            Some(b) => b,
            None => return false,
        };
        if self.defs.buildings[self.buildings[bi].def].shift != Shift::Day {
            return false;
        }
        if on && !self.extended.contains(&bid) {
            self.extended.push(bid);
        } else if !on {
            if let Some(p) = self.extended.iter().position(|&x| x == bid) {
                self.extended.remove(p);
            }
        }
        true
    }

    pub fn child_may_work_for(&self, s: &Survivor, bi: usize) -> bool {
        let d = self.buildings[bi].def;
        let min = self.defs.child_work_age.iter().find(|x| x.0 == d).map(|x| x.1).unwrap_or(self.defs.child_work_age_default);
        self.child_may_work(bi) && s.age >= min
    }

    pub fn child_may_work(&self, bi: usize) -> bool {
        self.has_law(self.defs.law_child) && self.defs.laws[self.defs.law_child.unwrap()].jobs.contains(&self.buildings[bi].def)
    }

    #[inline]
    pub fn survivor_by_id(&self, sid: i64) -> Option<usize> {
        if sid < 0 || sid as usize >= self.sidx.len() {
            return None;
        }
        let i = self.sidx[sid as usize];
        if i < 0 {
            None
        } else {
            Some(i as usize)
        }
    }

    // ============================================================ time

    #[inline]
    pub fn mpd(&self) -> i64 {
        self.defs.minutes_per_day as i64
    }
    #[inline]
    pub fn day(&self) -> i64 {
        self.minute / self.mpd()
    }
    #[inline]
    pub fn minute_of_day(&self) -> i64 {
        self.minute % self.mpd()
    }
    #[inline]
    pub fn hour(&self) -> i64 {
        self.minute_of_day() / 60
    }
    pub fn alive_count(&self) -> i64 {
        self.survivors.iter().filter(|s| s.alive()).count() as i64
    }

    // ============================================================ heat

    #[inline]
    pub fn outdoor_temp(&self) -> f64 {
        self.weather.outdoor_temp(&self.defs, self.minute)
    }
    #[inline]
    pub fn weather_level(&self) -> f64 {
        (self.outdoor_temp() - self.defs.zero_c) / self.defs.c_per_level
    }
    #[inline]
    pub fn in_zone(&self, p: V2) -> bool {
        let f = self.furnace();
        self.furnace_lit && self.buildings[f].level > 0 && (p.distance_to(self.center(f)) as f64) <= self.defs.furnace_radius
    }
    #[inline]
    pub fn zone_bonus(&self, p: V2) -> f64 {
        if self.in_zone(p) {
            self.furnace_heat()
        } else {
            0.0
        }
    }
    #[inline]
    pub fn ground_level(&self, p: V2) -> f64 {
        self.weather_level() + self.zone_bonus(p)
    }
    pub fn building_level(&self, bi: usize) -> f64 {
        let b = &self.buildings[bi];
        self.weather_level()
            + self.insulation_of(b.def)
            + self.zone_bonus(self.center(bi))
            + (if self.heater_running(bi) { b.heater as f64 } else { 0.0 })
    }
    #[inline]
    pub fn heater_running(&self, bi: usize) -> bool {
        self.buildings[bi].heater > 0 && self.heaters_fuelled
    }

    pub fn set_heater(&mut self, bid: i64, tier: i64) -> bool {
        let bi = match self.building_by_id(bid) {
            Some(b) => b,
            None => return false,
        };
        let d = self.buildings[bi].def;
        if fi(self.defs.buildings[d].beds) > 0 || d == self.furnace_def {
            return false;
        }
        self.buildings[bi].heater = tier.clamp(0, self.heater_max_level());
        true
    }

    pub fn fire_level(&self) -> f64 {
        let f = self.furnace();
        if !self.furnace_lit || self.buildings[f].level == 0 {
            return self.weather_level();
        }
        self.weather_level() + self.furnace_heat() + self.defs.fire_spot_bonus
    }

    #[inline]
    pub fn level_to_c(&self, level: f64) -> f64 {
        self.defs.zero_c + level * self.defs.c_per_level
    }

    fn env_level_for(&self, si: usize) -> f64 {
        let s = &self.survivors[si];
        if s.place_id >= 0 {
            if let Some(bi) = self.building_by_id(s.place_id) {
                if self.buildings[bi].built {
                    return self.building_level(bi);
                }
            }
        }
        if matches!(s.act, Act::Warm | Act::Sleep | Act::Rest | Act::Idle) && (s.pos.distance_to(self.fire_spot(s.id)) as f64) < 0.05 {
            return self.fire_level();
        }
        self.ground_level(s.pos)
    }

    fn update_heaters(&mut self) {
        let mut need = 0.0f64;
        let minus = self.tech_sum_heater_coal_minus();
        for b in &self.buildings {
            if b.heater <= 0 {
                continue;
            }
            need += maxf(b.heater as f64 - minus, 0.0) * self.defs.heater_coal_per_level_hour * self.defs.heater_building_scale / 60.0;
        }
        if need <= 0.0 {
            self.heaters_fuelled = true;
            return;
        }
        self.heaters_fuelled = self.res[0] >= need;
        if self.heaters_fuelled {
            self.res[0] -= need;
        }
    }

    pub fn update_furnace(&mut self) {
        let f = self.furnace();
        let level = self.buildings[f].level;
        let need = level as f64 * self.defs.furnace_coal_per_level_hour * self.defs.k * self.tech_product_furnace_coal_mult() / 60.0;
        if level == 0 {
            self.furnace_lit = false;
            return;
        }
        let have = if self.furnace_lit { need } else { need * self.defs.relight_minutes };
        if self.res[0] >= have {
            self.res[0] -= need;
            self.furnace_lit = true;
        } else {
            self.furnace_lit = false;
        }
    }

    // ============================================================ tick

    pub fn tick(&mut self) {
        self.minute += 1;
        self.update_furnace();
        self.update_pressure();
        self.update_heaters();
        self.weather_events();
        let m = self.minute_of_day();
        if m == 0 {
            self.daily();
        }
        if m == fi(self.defs.sickness_roll_hour) * 60 {
            self.sickness_roll();
        }
        if m == fi(self.defs.meal_hour) * 60 {
            self.serve_meals();
        }
        self.update_expeditions();
        for si in 0..self.survivors.len() {
            let s = &self.survivors[si];
            if s.alive() && s.job_id != SCOUT_JOB {
                self.decide(si);
                self.do_move(si);
                self.body(si);
                if self.survivors[si].alive() {
                    self.illness(si);
                }
                if self.survivors[si].alive() {
                    self.hunger(si);
                }
            }
        }
        self.produce();
        self.research();
    }

    pub fn advance(&mut self, minutes: i64) {
        for _ in 0..minutes {
            self.tick();
        }
    }

    /// GDD 7: real seconds away -> game minutes at 1/4 online speed, capped.
    pub fn offline_minutes(&self, real_seconds: f64) -> i64 {
        let secs = clampf(real_seconds, 0.0, self.defs.off_max_real_hours * 3600.0);
        fi(secs * self.defs.off_game_days_per_real_day / 60.0)
    }

    /// Catch up time away; nobody dies offline, overdrive is switched off.
    pub fn advance_offline(&mut self, minutes: i64) {
        self.overdrive = false;
        self.offline = true;
        self.advance(minutes);
        self.offline = false;
    }

    fn weather_events(&mut self) {
        let day_len = self.mpd();
        let lead = fi(self.defs.forecast_days) * day_len;
        if self.weather.chaotic {
            if self.minute % 60 == 0 {
                self.weather.hourly(&self.defs, self.minute);
            }
            if self.minute == self.weather.next_storm_start {
                self.hope = clampf(self.hope + self.defs.storm_forecast_hope, 0.0, 100.0);
            }
        } else if self.minute == self.weather.next_storm_start - lead {
            self.hope = clampf(self.hope + self.defs.storm_forecast_hope, 0.0, 100.0);
        }
        if self.minute == self.weather.next_storm_end {
            let won = self.weather.next_storm_scenario_end && self.alive_count() > 0 && !self.scenario_won;
            self.weather.storm_ended(&self.defs);
            if won {
                self.scenario_won = true;
                self.chapter = (self.chapter + 1).min(self.defs.n_chapters - 1);
            }
        }
        if self.minute_of_day() == fi(self.defs.weather_change_hour) * 60 {
            let d = self.day();
            self.weather.daily_change(&self.defs, &mut self.rng, d);
        }
    }

    // ============================================================ decisions

    fn on_shift(&self, bi: usize) -> bool {
        let h = self.hour();
        let d = &self.defs;
        match d.buildings[self.buildings[bi].def].shift {
            Shift::Always => true,
            Shift::Night => h >= fi(d.hunt_start_hour) || h < fi(d.hunt_end_hour),
            Shift::Day => {
                if self.extended.contains(&self.buildings[bi].id) {
                    let ex = &d.laws[d.law_extended.unwrap()];
                    return h >= fi(ex.start_hour) && h < fi(ex.end_hour);
                }
                h >= fi(d.work_start_hour) && h < fi(d.work_end_hour)
            }
        }
    }

    pub fn operable(&self, bi: usize) -> bool {
        let d = &self.defs.buildings[self.buildings[bi].def];
        if !self.buildings[bi].built {
            return true;
        }
        if let Some(ml) = d.min_level {
            if self.building_level(bi) < ml {
                return false;
            }
        }
        if d.storm_stops && self.weather.in_storm(self.minute) {
            return false;
        }
        if self.storm_shelter && self.weather.in_storm(self.minute) && fi(d.insulation) == 0 {
            return false;
        }
        true
    }

    fn workplace_for(&self, si: usize) -> Option<usize> {
        let s = &self.survivors[si];
        if s.job_id == BUILDER_JOB && s.role != CHILD {
            let h = self.hour();
            if h < fi(self.defs.work_start_hour) || h >= fi(self.defs.work_end_hour) {
                return None;
            }
            return self.buildings.iter().position(|b| !b.built);
        }
        if s.job_id >= 0 {
            if let Some(bi) = self.building_by_id(s.job_id) {
                if self.buildings[bi].built && self.on_shift(bi) && self.operable(bi) {
                    return Some(bi);
                }
            }
        }
        None
    }

    #[inline]
    fn collapse_at(&self) -> f64 {
        self.defs.hypo_stage_work[self.defs.hypo_stage_work.len() - 1].0
    }

    fn decide(&mut self, si: usize) {
        let collapse = self.collapse_at();
        {
            let s = &mut self.survivors[si];
            if s.act == Act::Collapsed || s.hypo >= collapse {
                s.act = Act::Collapsed;
                s.place_id = -1;
                if s.hypo < collapse {
                    s.act = Act::Idle;
                }
                return;
            }
        }
        let mut act = Act::Sleep;
        let s = &self.survivors[si];
        let mut dest_id = s.bed_id;
        if s.illness != NONE {
            act = Act::Rest;
            dest_id = match self.free_care_bed(si) {
                Some(bi) => self.buildings[bi].id,
                None => s.bed_id,
            };
        } else if s.hypo >= self.defs.warm_up_from || (s.act == Act::Warm && s.hypo > self.defs.warm_up_until) {
            act = Act::Warm;
            dest_id = -1;
        } else if self.minute < s.break_until {
            // RW mental break: no work
        } else if let Some(wp) = self.workplace_for(si) {
            act = if self.defs.buildings[self.buildings[wp].def].shift == Shift::Night { Act::Away } else { Act::Work };
            dest_id = self.buildings[wp].id;
        }
        let sid = s.id;
        let mut dest = self.fire_spot(sid);
        if dest_id >= 0 {
            match self.building_by_id(dest_id) {
                None => dest_id = -1,
                Some(bi) => {
                    dest = if self.buildings[bi].built && self.is_shelter_like(bi) { self.center(bi) } else { self.work_spot(bi, sid) };
                }
            }
        }
        let s = &mut self.survivors[si];
        let here = s.act != Act::Walk && s.place_id == dest_id && (s.pos.distance_to(dest) as f64) < 0.05;
        if here {
            s.act = act;
            return;
        }
        if s.act == Act::Walk && s.dest_place_id == dest_id && s.target == dest {
            s.next_act = act;
            return;
        }
        s.act = Act::Walk;
        s.next_act = act;
        s.target = dest;
        s.dest_place_id = dest_id;
        s.place_id = -1;
    }

    fn is_shelter_like(&self, bi: usize) -> bool {
        let d = &self.defs.buildings[self.buildings[bi].def];
        fi(d.beds) > 0 || d.care_beds.is_some() || d.has_consumes
    }

    fn free_care_bed(&self, si: usize) -> Option<usize> {
        for (bi, b) in self.buildings.iter().enumerate() {
            let d = &self.defs.buildings[b.def];
            // FP: without Radical Treatment/Sustain Life a Medical Post does not take the gravely ill.
            if b.built && (self.survivors[si].illness != GRAVE || d.cure_hours_grave.is_some()) {
                if let Some(cb) = d.care_beds {
                    let mut used = 0i64;
                    for (oi, o) in self.survivors.iter().enumerate() {
                        if oi != si && o.alive() && o.illness != NONE && (o.place_id == b.id || o.dest_place_id == b.id) {
                            used += 1;
                        }
                    }
                    if used < fi(cb) {
                        return Some(bi);
                    }
                }
            }
        }
        None
    }

    fn do_move(&mut self, si: usize) {
        if self.survivors[si].act != Act::Walk {
            return;
        }
        let mut speed = self.defs.walk_tiles_per_minute * stage_mult(&self.defs.hypo_stage_move, self.survivors[si].hypo);
        let s = &mut self.survivors[si];
        if s.illness != NONE {
            speed *= 0.5;
        }
        let to = s.target.sub(s.pos);
        if (to.length() as f64) <= speed {
            s.pos = s.target;
            s.act = s.next_act;
            s.place_id = s.dest_place_id;
        } else {
            s.pos = s.pos.add(to.normalized().mul(speed));
        }
    }

    pub fn work_efficiency(&self, s: &Survivor) -> f64 {
        let mut e = stage_mult(&self.defs.hypo_stage_work, s.hypo);
        e *= maxf(0.0, 1.0 - self.defs.frostbite_work_loss * s.lost_digits().min(self.defs.frostbite_fingers) as f64);
        if s.pregnant_since >= 0 {
            let frac = (self.minute - s.pregnant_since) as f64 / (self.defs.gestation_days * self.defs.minutes_per_day);
            for st in &self.defs.pregnancy_work {
                if frac >= st.0 {
                    e *= st.1;
                    break;
                }
            }
        }
        if s.gear[2] >= 0 {
            e *= 1.0 + self.defs.gear_tool_work_bonus * self.defs.gear_tool_mult[s.gear[2] as usize];
        }
        e
    }

    /// RW: worn clothing lowers the safe and comfortable temperatures by its insulation.
    pub fn insulation_c(&self, s: &Survivor) -> f64 {
        let mut total = 0.0f64;
        for k in 0..2 {
            if s.gear[k] >= 0 {
                total += self.defs.gear_kind_insulation[k] * self.defs.gear_insulation_mult[s.gear[k] as usize];
            }
        }
        total
    }

    // ============================================================ body

    fn body(&mut self, si: usize) {
        if self.survivors[si].act == Act::Away {
            return;
        }
        let env = self.env_level_for(si);
        let ins = self.insulation_c(&self.survivors[si]);
        let d = &self.defs;
        let s = &mut self.survivors[si];
        let diff = env - s.body_level;
        if diff.abs() > 0.0001 {
            let table = &d.change_hours_by_diff;
            let mut hours = table[((diff.abs().ceil() as i64) - 1).clamp(0, table.len() as i64 - 1) as usize];
            if diff > 0.0 && s.body_level >= 1.0 && diff.abs() <= 1.0 {
                hours = d.livable_to_comfortable_hours;
            }
            s.body_level += signf(diff) * minf(diff.abs(), 1.0 / (hours * 60.0));
        }
        let eff_c = d.zero_c + env * d.c_per_level;
        let safe = d.hypo_safe_c - ins;
        if eff_c < safe {
            s.hypo += maxf(d.hypo_min_rate_h, d.hypo_rate_per_c_h * (safe - eff_c)) / 60.0;
        } else if env + ins / d.c_per_level >= d.hypo_recover_level && s.hypo > 0.0 {
            s.hypo -= clampf(d.hypo_recover_factor_h * s.hypo, d.hypo_recover_min_h, d.hypo_recover_max_h) / 60.0;
        }
        s.hypo = clampf(s.hypo, 0.0, 1.0);
        if eff_c < 0.0 && s.hypo > d.frostbite_min_sev && s.lost_digits() < d.frostbite_digits {
            let p = 1.0 - (1.0 - d.frostbite_chance_h).powf(1.0 / 60.0);
            if self.rng.randf() < p {
                s.frostbites += 1;
            }
        }
        // FP has no direct cold death: cold kills only through illness.
    }

    // ============================================================ sickness

    fn sickness_roll(&mut self) {
        let d = &self.defs;
        for s in self.survivors.iter_mut() {
            if !s.alive() || s.illness != NONE || s.act == Act::Away {
                continue;
            }
            let lv = (s.body_level.floor() as i64).clamp(-3, 2);
            let target = d.sick_target[(lv + 3) as usize];
            s.sick_chance += (target - s.sick_chance) * d.sick_chance_approach_per_day;
            if self.rng.randf() < s.sick_chance {
                s.illness = SICK;
                s.illness_days = 0.0;
                if lv <= fi(d.gravely_ill_from_level) && self.rng.randf() < d.gravely_ill_chance {
                    s.illness = GRAVE;
                }
            }
        }
    }

    fn illness(&mut self, si: usize) {
        if self.survivors[si].illness == NONE {
            return;
        }
        let mut treat = 0.0;
        let mut hours = 1.0;
        let s = &self.survivors[si];
        if s.place_id >= 0 {
            if let Some(bi) = self.building_by_id(s.place_id) {
                let d = &self.defs.buildings[self.buildings[bi].def];
                let h = if s.illness == SICK { d.cure_hours_sick } else { d.cure_hours_grave };
                if let (Some(_), Some(h)) = (d.care_beds, h) {
                    hours = h;
                    treat = self.staff_fraction(bi);
                }
            }
        }
        if treat > 0.0 {
            let cm = self.tech_product_cure_mult();
            let s = &mut self.survivors[si];
            s.cure_progress += treat * cm / (hours * 60.0);
            if s.cure_progress >= 1.0 {
                s.illness = NONE;
                s.illness_days = 0.0;
                s.cure_progress = 0.0;
            }
            return;
        }
        let stage = self.defs.body_stage_days;
        let s = &mut self.survivors[si];
        s.illness_days += 1.0 / 1440.0;
        if s.illness_days >= stage {
            if s.illness == SICK {
                s.illness = GRAVE;
                s.illness_days = 0.0;
            } else {
                self.lethal(si, Cause::Sick);
            }
        }
    }

    fn staff_fraction(&self, bi: usize) -> f64 {
        let cap = fi(self.defs.buildings[self.buildings[bi].def].workers);
        let bid = self.buildings[bi].id;
        let n = self.survivors.iter().filter(|o| o.alive() && o.job_id == bid && o.act == Act::Work && o.place_id == bid).count();
        n as f64 / cap.max(1) as f64
    }

    // ============================================================ food

    fn hunger(&mut self, si: usize) {
        let s = &mut self.survivors[si];
        s.unfed_days += 1.0 / 1440.0;
        if s.unfed_days >= 3.0 * self.defs.food_stage_days {
            self.lethal(si, Cause::Starve);
        }
    }

    #[inline]
    pub fn hunger_stage(&self, s: &Survivor) -> i64 {
        fi(s.unfed_days / self.defs.food_stage_days)
    }

    fn serve_meals(&mut self) {
        let stage_days = self.defs.food_stage_days;
        let additives = self.has_law(self.defs.law_additives);
        let add_chance = self.defs.law_additives.map(|l| self.defs.laws[l].sick_chance).unwrap_or(0.0);
        for si in 0..self.survivors.len() {
            let hs = self.hunger_stage(&self.survivors[si]);
            let s = &mut self.survivors[si];
            if !s.alive() || s.job_id == SCOUT_JOB {
                continue;
            }
            let need = if hs >= 2 { self.defs.starving_rations } else { self.defs.rations_per_day };
            if self.res[4] >= need {
                self.res[4] -= need;
                s.unfed_days = maxf(0.0, s.unfed_days - stage_days);
                if additives && s.illness == NONE && self.rng.randf() < add_chance {
                    s.illness = SICK;
                    s.illness_days = 0.0;
                }
            } else if self.res[3] >= need {
                self.res[3] -= need;
                s.unfed_days = maxf(0.0, s.unfed_days - stage_days);
                if s.illness == NONE && self.rng.randf() < self.defs.raw_sick_chance {
                    s.illness = SICK;
                    s.illness_days = 0.0;
                }
            }
        }
    }

    // ============================================================ death

    fn lethal(&mut self, si: usize, cause: Cause) {
        if !self.survivors[si].alive() {
            return;
        }
        if self.offline || self.difficulty == GENTLE {
            let s = &mut self.survivors[si];
            match cause {
                Cause::Sick => s.illness_days = self.defs.body_stage_days - 0.001,
                Cause::Starve => s.unfed_days = 3.0 * self.defs.food_stage_days - 0.001,
            }
            return;
        }
        self.die(si);
    }

    fn die(&mut self, si: usize) {
        self.drop_gear(si);
        let day = self.day();
        let s = &mut self.survivors[si];
        s.act = Act::Dead;
        s.died_day = day;
        s.place_id = -1;
        s.bed_id = -1;
        s.job_id = -1;
        self.hope = clampf(self.hope + self.defs.death_hope, 0.0, 100.0);
        self.discontent = clampf(self.discontent + self.defs.death_discontent, 0.0, 100.0);
        self.auto_assign_beds();
        self.auto_equip();
    }

    pub fn auto_assign_beds(&mut self) {
        let mut beds: Vec<(f64, i64, usize)> = Vec::new();
        for (bi, b) in self.buildings.iter().enumerate() {
            let d = &self.defs.buildings[b.def];
            if b.built && fi(d.beds) > 0 {
                beds.push((self.insulation_of(b.def) + self.zone_potential(self.center(bi)), b.id, bi));
            }
        }
        // total order (key desc, id asc), so any correct sort gives Godot's result
        beds.sort_by(|a, c| {
            if a.0 != c.0 {
                c.0.partial_cmp(&a.0).unwrap()
            } else {
                a.1.cmp(&c.1)
            }
        });
        let mut people: Vec<(i64, i64, usize)> = Vec::new();
        for (si, s) in self.survivors.iter_mut().enumerate() {
            s.bed_id = -1;
            if s.alive() {
                people.push((0, s.id, si));
            }
        }
        for p in people.iter_mut() {
            p.0 = bed_priority(&self.survivors[p.2]);
        }
        people.sort_by(|a, c| if a.0 != c.0 { c.0.cmp(&a.0) } else { a.1.cmp(&c.1) });
        let mut i = 0usize;
        for &(_, _, bi) in &beds {
            let b = &self.buildings[bi];
            for _ in 0..fi(self.defs.buildings[b.def].beds) {
                if i < people.len() {
                    self.survivors[people[i].2].bed_id = b.id;
                    i += 1;
                }
            }
        }
    }

    pub fn zone_potential(&self, p: V2) -> f64 {
        if (p.distance_to(self.center(self.furnace())) as f64) <= self.defs.furnace_radius {
            1.0
        } else {
            0.0
        }
    }

    // ============================================================ economy

    fn produce(&mut self) {
        let mut depleted: Vec<i64> = Vec::new();
        let n = self.next_id as usize + 1;
        let mut eff_at: Vec<f64> = vec![0.0; n];
        let mut has_eff: Vec<bool> = vec![false; n];
        for s in &self.survivors {
            if s.alive() && s.place_id >= 0 && (s.act == Act::Work || s.act == Act::Away) {
                let p = s.place_id as usize;
                eff_at[p] = (if has_eff[p] { eff_at[p] } else { 0.0 }) + self.work_efficiency(s);
                has_eff[p] = true;
            }
        }
        for bi in 0..self.buildings.len() {
            let bid = self.buildings[bi].id as usize;
            let dix = self.buildings[bi].def;
            let mut eff = if has_eff[bid] { eff_at[bid] } else { 0.0 };
            if eff <= 0.0 {
                continue;
            }
            let d = &self.defs.buildings[dix];
            if self.buildings[bi].built && d.researches {
                continue;
            }
            if !self.buildings[bi].built {
                let b = &mut self.buildings[bi];
                b.progress += minf(eff, 1.0);
                if b.progress >= d.build_minutes {
                    b.built = true;
                    self.auto_assign_beds();
                }
                continue;
            }
            eff *= self.rate_mult(dix);
            let d = &self.defs.buildings[dix];
            if let Some((pres, _)) = d.pile {
                let b = &mut self.buildings[bi];
                let take = minf(b.amount, d.rate * eff / 60.0);
                b.amount -= take;
                self.res[pres] += take;
                if b.amount <= 0.0 {
                    depleted.push(b.id);
                }
            } else if let Some(radius) = d.gathers_piles_radius {
                if let Some(pi) = self.nearest_pile(bi, radius) {
                    let pres = self.defs.buildings[self.buildings[pi].def].pile.unwrap().0;
                    let rate = self.defs.buildings[dix].gather_rates[pres];
                    let pile = &mut self.buildings[pi];
                    let take = minf(pile.amount, rate * eff / 60.0);
                    pile.amount -= take;
                    self.res[pres] += take;
                    if pile.amount <= 0.0 {
                        depleted.push(pile.id);
                    }
                }
            } else if let Some(radius) = d.harvests_forest_radius {
                let idx = self.nearest_forest(bi, radius);
                if idx >= 0 {
                    let idx = idx as usize;
                    let take = minf(self.tile_amount[idx], self.defs.buildings[dix].rate * eff / 60.0);
                    self.tile_amount[idx] -= take;
                    self.res[1] += take;
                    if self.tile_amount[idx] <= 0.0 {
                        self.tiles[idx] = SNOW;
                    }
                }
            } else {
                let mut factor = 1.0f64;
                for &(k, c) in &d.consumes {
                    let need = c / 60.0 * eff;
                    factor = minf(factor, if need > 0.0 { self.res[k] / need } else { 1.0 });
                }
                factor = clampf(factor, 0.0, 1.0);
                for &(k, c) in &d.consumes {
                    self.res[k] -= c / 60.0 * eff * factor;
                }
                let recipe = if d.produces.iter().any(|p| p.0 == 4) { self.ration_multiplier() } else { 1.0 };
                for &(k, p) in &d.produces {
                    self.res[k] += p / 60.0 * eff * factor * recipe;
                }
            }
        }
        for pid in depleted {
            if let Some(bi) = self.building_by_id(pid) {
                self.mark(bi, -1);
                self.buildings.remove(bi);
                if let Some(p) = self.extended.iter().position(|&x| x == pid) {
                    self.extended.remove(p);
                }
                self.reindex();
                for s in self.survivors.iter_mut() {
                    if s.job_id == pid {
                        s.job_id = -1;
                    }
                }
            }
        }
    }

    pub fn ration_multiplier(&self) -> f64 {
        for l in [self.defs.law_additives, self.defs.law_soup] {
            if self.has_law(l) {
                let l = &self.defs.laws[l.unwrap()];
                return l.rations_per_batch / l.raw_per_batch / 2.0;
            }
        }
        1.0
    }

    pub fn nearest_pile(&self, post: usize, radius: f64) -> Option<usize> {
        let mut best = None;
        let mut best_d = f64::INFINITY;
        let pc = self.center(post);
        for (bi, b) in self.buildings.iter().enumerate() {
            if self.defs.buildings[b.def].pile.is_some() && b.amount > 0.0 {
                let dd = self.center(bi).distance_to(pc) as f64;
                if dd <= radius && dd < best_d {
                    best_d = dd;
                    best = Some(bi);
                }
            }
        }
        best
    }

    fn nearest_forest(&self, bi: usize, radius: f64) -> i64 {
        let c = self.center(bi);
        let mut best = -1i64;
        let mut best_d = f64::INFINITY;
        let (cx, cy) = (c.x as f64, c.y as f64);
        for y in (fi(cy - radius) - 1)..(fi(cy + radius) + 2) {
            for x in (fi(cx - radius) - 1)..(fi(cx + radius) + 2) {
                if !self.in_bounds(x, y) {
                    continue;
                }
                let idx = y * self.map_size + x;
                if self.tiles[idx as usize] == FOREST && self.tile_amount[idx as usize] > 0.0 {
                    let dd = V2::f(x as f64 + 0.5, y as f64 + 0.5).distance_to(c) as f64;
                    if dd <= radius && dd < best_d {
                        best_d = dd;
                        best = idx;
                    }
                }
            }
        }
        best
    }

    // ============================================================ scouting

    pub fn scouting_available(&self) -> bool {
        match self.first_building(self.defs.bix_beacon()) {
            Some(bi) => self.buildings[bi].built && (self.teams.len() as i64) < fi(self.defs.max_teams).max(fi(self.tech_max_teams())),
            None => false,
        }
    }

    pub fn send_expedition(&mut self, site_index: i64, member_ids: &[i64]) -> bool {
        if !self.scouting_available() || site_index < 0 || site_index >= self.sites.len() as i64 || self.sites[site_index as usize].visited {
            return false;
        }
        if self.teams.iter().any(|t| t.site == site_index) {
            return false;
        }
        if member_ids.len() as i64 != fi(self.defs.team_members) {
            return false;
        }
        let mut members = Vec::new();
        for &sid in member_ids {
            let si = match self.survivor_by_id(sid) {
                Some(i) => i,
                None => return false,
            };
            let s = &self.survivors[si];
            if !s.alive() || s.illness != NONE || s.job_id == SCOUT_JOB {
                return false;
            }
            if !self.defs.team_allowed[s.role as usize] {
                return false;
            }
            members.push(si);
        }
        for &(k, c) in &self.defs.team_cost {
            if self.res[k] < c {
                return false;
            }
        }
        for &(k, c) in &self.defs.team_cost {
            self.res[k] -= c;
        }
        let beacon_pos = self.center(self.first_building(self.defs.bix_beacon()).unwrap());
        let mut ids = Vec::new();
        for si in members {
            let s = &mut self.survivors[si];
            s.job_id = SCOUT_JOB;
            s.act = Act::Away;
            s.place_id = -1;
            s.pos = beacon_pos;
            s.target = beacon_pos;
            ids.push(s.id);
        }
        let eta = self.minute + fi(self.sites[site_index as usize].hours * 60.0 / self.scout_speed());
        self.teams.push(Team { members: ids, site: site_index, back: false, eta });
        true
    }

    fn update_expeditions(&mut self) {
        let storm = self.weather.in_storm(self.minute);
        let snapshot = self.teams.clone();
        for team in snapshot {
            if storm {
                if self.offline {
                    if let Some(t) = self.teams.iter_mut().find(|t| **t == team) {
                        t.eta += 1;
                    }
                    continue;
                }
                self.erase_team(&team);
                for &sid in &team.members {
                    self.kill_now(sid);
                }
                continue;
            }
            if self.minute < team.eta {
                continue;
            }
            let site = team.site as usize;
            let sd = self.sites[site].def;
            if !team.back {
                self.sites[site].visited = true;
                if self.defs.sites[sd].risky && !self.offline && self.rng.randf() < self.defs.risky_loss_chance {
                    self.erase_team(&team);
                    for &sid in &team.members {
                        self.kill_now(sid);
                    }
                    continue;
                }
                for i in 0..self.defs.sites[sd].loot.len() {
                    let (k, v) = self.defs.sites[sd].loot[i];
                    self.res[k] = self.res[k] + v;
                }
                let mut roles = Vec::new();
                for i in 0..self.defs.sites[sd].people.len() {
                    let (role, x) = self.defs.sites[sd].people[i];
                    let n = fi(x.floor()) + if self.rng.randf() < x - x.floor() { 1 } else { 0 };
                    for _ in 0..n {
                        roles.push(role);
                    }
                }
                for _ in 0..fi(self.defs.gear_items_per_site) {
                    self.loot_gear(sd);
                }
                let hours = self.sites[site].hours;
                if !roles.is_empty() {
                    self.incoming.push(Group { arrive: self.minute + fi(hours * 60.0 * self.defs.group_walk_factor), roles });
                }
                let eta = self.minute + fi(hours * 60.0 * self.defs.return_factor / self.scout_speed());
                if let Some(t) = self.teams.iter_mut().find(|t| **t == team) {
                    t.back = true;
                    t.eta = eta;
                }
            } else {
                self.erase_team(&team);
                let at = match self.first_building(self.defs.bix_beacon()) {
                    Some(bi) => self.center(bi),
                    None => self.fire_spot(0),
                };
                for &sid in &team.members {
                    if let Some(si) = self.survivor_by_id(sid) {
                        let s = &mut self.survivors[si];
                        if s.alive() {
                            s.job_id = -1;
                            s.act = Act::Idle;
                            s.pos = at;
                            s.target = at;
                        }
                    }
                }
            }
        }
        let groups = self.incoming.clone();
        for g in groups {
            if self.minute >= g.arrive {
                if let Some(p) = self.incoming.iter().position(|x| *x == g) {
                    self.incoming.remove(p);
                }
                for &r in &g.roles {
                    let at = self.fire_spot(self.survivors.len() as i64);
                    self.add_survivor(at, r);
                }
                self.auto_assign_beds();
                self.hope = clampf(self.hope + 2.0, 0.0, 100.0);
            }
        }
    }

    /// One gear item: kind uniform, RW quality table (BaseGen / Reward), DD long-site floor +1.
    fn loot_gear(&mut self, sd: usize) {
        let kind = self.rng.randi_range(0, self.defs.gear_kind_insulation.len() as i64 - 1);
        let risky = self.defs.sites[sd].risky;
        let x = self.rng.randf();
        let table = if risky { &self.defs.gear_table_risky } else { &self.defs.gear_table_normal };
        let mut acc = 0.0f64;
        let mut q: i64 = -1;
        for (i, &p) in table.iter().enumerate() {
            acc += p;
            if p > 0.0 {
                q = i as i64;
                if x < acc {
                    break;
                }
            }
        }
        if self.defs.sites[sd].hours[0] >= self.defs.gear_long_site_hours {
            q = (q + 1).min(table.len() as i64 - 1);
        }
        self.gear_stock.push((kind, q));
        self.auto_equip();
    }

    pub fn auto_equip(&mut self) {
        for kind in 0..3usize {
            loop {
                let mut best: i64 = -1;
                for (i, it) in self.gear_stock.iter().enumerate() {
                    if it.0 == kind as i64 && (best < 0 || it.1 > self.gear_stock[best as usize].1) {
                        best = i as i64;
                    }
                }
                if best < 0 {
                    break;
                }
                let mut who: Option<usize> = None;
                for (si, s) in self.survivors.iter().enumerate() {
                    if !s.alive() || s.job_id == SCOUT_JOB {
                        continue;
                    }
                    match who {
                        None => who = Some(si),
                        Some(wi) => {
                            let w = &self.survivors[wi];
                            if s.gear[kind] < w.gear[kind] || (s.gear[kind] == w.gear[kind] && w.role == CHILD && s.role != CHILD) {
                                who = Some(si);
                            }
                        }
                    }
                }
                let q = self.gear_stock[best as usize].1;
                let wi = match who {
                    Some(wi) if self.survivors[wi].gear[kind] < q => wi,
                    _ => break,
                };
                self.gear_stock.remove(best as usize);
                let old = self.survivors[wi].gear[kind];
                if old >= 0 {
                    self.gear_stock.push((kind as i64, old));
                }
                self.survivors[wi].gear[kind] = q;
            }
        }
    }

    fn drop_gear(&mut self, si: usize) {
        for k in 0..3 {
            let s = &self.survivors[si];
            if s.gear[k] >= 0 && s.job_id != SCOUT_JOB {
                self.gear_stock.push((k as i64, s.gear[k]));
                self.survivors[si].gear[k] = -1;
            }
        }
    }

    fn erase_team(&mut self, team: &Team) {
        if let Some(p) = self.teams.iter().position(|t| t == team) {
            self.teams.remove(p);
        }
    }

    fn kill_now(&mut self, sid: i64) {
        let si = match self.survivor_by_id(sid) {
            Some(i) => i,
            None => return,
        };
        if !self.survivors[si].alive() {
            return;
        }
        if self.offline {
            let s = &mut self.survivors[si];
            s.job_id = -1;
            s.act = Act::Idle;
            return;
        }
        self.drop_gear(si);
        let day = self.day();
        let s = &mut self.survivors[si];
        s.act = Act::Dead;
        s.died_day = day;
        s.job_id = -1;
        s.bed_id = -1;
        self.hope = clampf(self.hope + self.defs.death_hope, 0.0, 100.0);
        self.discontent = clampf(self.discontent + self.defs.death_discontent, 0.0, 100.0);
        self.auto_assign_beds();
        self.auto_equip();
    }

    // ============================================================ daily

    fn daily(&mut self) {
        let (mut homeless, mut cold_home, mut hungry) = (false, false, false);
        for si in 0..self.survivors.len() {
            let s = &self.survivors[si];
            if !s.alive() {
                continue;
            }
            if s.bed_id < 0 {
                homeless = true;
            } else if let Some(bi) = self.building_by_id(s.bed_id) {
                if self.building_level(bi) < 0.0 {
                    cold_home = true;
                }
            }
            if self.hunger_stage(&self.survivors[si]) >= 1 {
                hungry = true;
            }
        }
        let n = homeless as i64 + cold_home as i64 + hungry as i64 + (!self.furnace_lit) as i64;
        let target = n as f64 * self.defs.needs_discontent_tiers * self.defs.laws_points_per_tier;
        self.discontent += target - self.needs_discontent;
        self.needs_discontent = target;
        if let Some(le) = self.defs.law_extended {
            let per = self.defs.laws[le].discontent_per_building_day;
            for &bid in &self.extended {
                if self.survivors.iter().any(|s| s.alive() && s.job_id == bid) {
                    self.discontent += per;
                }
            }
        }
        self.discontent += self.defs.daily_discontent_drift;
        if self.hope < self.defs.start_hope {
            self.hope += self.defs.daily_hope_drift;
        }
        self.hope = clampf(self.hope, 0.0, 100.0);
        self.discontent = clampf(self.discontent, 0.0, 100.0);
        self.mental_breaks();
        self.ultimatum_check();
        self.life();
        self.population_floor();
        self.auto_equip();
        self.roll_events();
    }

    fn mental_breaks(&mut self) {
        let mut tier: Option<usize> = None;
        for (i, t) in self.defs.breaks.iter().enumerate() {
            if self.hope < t.0 {
                tier = Some(i);
            }
        }
        let t = match tier {
            Some(i) => self.defs.breaks[i],
            None => return,
        };
        let p = 1.0 - (-1.0 / t.1).exp();
        let calm = fi(self.defs.catharsis_days * self.defs.minutes_per_day);
        for si in 0..self.survivors.len() {
            let s = &self.survivors[si];
            if !s.alive() || s.job_id == SCOUT_JOB || self.minute < s.calm_until {
                continue;
            }
            if self.rng.randf() < p {
                let h = self.rng.randf_range(t.2, t.3);
                let s = &mut self.survivors[si];
                s.break_until = self.minute + fi(h * 60.0);
                s.calm_until = s.break_until + calm;
            }
        }
    }

    fn ultimatum_check(&mut self) {
        if self.exiled || self.difficulty == GENTLE {
            return;
        }
        let d = &self.defs;
        let day_len = self.mpd();
        if self.ultimatum == 0 {
            if self.discontent >= d.ult_discontent_at {
                self.ultimatum = 1;
            } else if self.hope <= d.ult_hope_at {
                self.ultimatum = 2;
            } else {
                return;
            }
            self.ultimatum_end = self.minute + fi(d.ult_days) * day_len;
            return;
        }
        let cleared = if self.ultimatum == 1 { self.discontent < d.ult_clear_discontent } else { self.hope > d.ult_clear_hope };
        if cleared {
            self.ultimatum = 0;
            self.ultimatum_end = -1;
            self.hope = clampf(self.hope + d.ult_success_hope_tiers * d.laws_points_per_tier, 0.0, 100.0);
        } else if self.minute >= self.ultimatum_end {
            if self.offline {
                self.ultimatum_end += day_len;
                return;
            }
            self.ultimatum = 0;
            self.ultimatum_end = -1;
            self.exiled = true;
        }
    }

    pub fn is_female(&self, s: &Survivor) -> bool {
        FEMALE[((s.id * 7) % N_NAMES) as usize] == 1
    }

    fn single_adult(&self, si: usize) -> bool {
        let s = &self.survivors[si];
        if !s.alive() || s.role == CHILD || s.age < self.defs.partner_min_age {
            return false;
        }
        match self.survivor_by_id(s.partner_id) {
            None => true,
            Some(pi) => !self.survivors[pi].alive(),
        }
    }

    fn life(&mut self) {
        let mpd = self.mpd();
        let n = self.survivors.len();
        for i in 0..n {
            if !self.survivors[i].alive() {
                continue;
            }
            let d = &self.defs;
            let s = &mut self.survivors[i];
            let mut rate = 1.0f64;
            if (s.age as f64) < d.child_rate_until {
                rate = d.child_rate;
            } else if (s.age as f64) < d.adult_rate_from {
                rate = d.child_rate + (1.0 - d.child_rate) * (s.age as f64 - d.child_rate_until) / (d.adult_rate_from - d.child_rate_until);
            }
            s.age_days += rate;
            if s.age_days >= d.days_per_year {
                s.age_days -= d.days_per_year;
                s.age += 1;
                if s.role == CHILD && s.age >= d.adult_age {
                    s.role = WORKER;
                    s.job_id = -1;
                }
                let p = curve(&d.heart_attack_year, s.age as f64);
                if p > 0.0 && self.rng.randf() < p {
                    if self.rng.randf() < self.defs.heart_attack_death && !self.offline && self.difficulty != GENTLE {
                        self.die(i);
                    }
                }
            }
        }
        for i in 0..self.survivors.len() {
            if !self.single_adult(i) || !self.is_female(&self.survivors[i]) {
                continue;
            }
            let s = &self.survivors[i];
            let mut best: Option<usize> = None;
            for (oi, o) in self.survivors.iter().enumerate() {
                if oi != i && self.single_adult(oi) && !self.is_female(o) && !s.parents.contains(&o.id) && !o.parents.contains(&s.id) {
                    match best {
                        None => best = Some(oi),
                        Some(bi) => {
                            if (o.age - s.age).abs() < (self.survivors[bi].age - s.age).abs() {
                                best = Some(oi);
                            }
                        }
                    }
                }
            }
            if let Some(bi) = best {
                let (sid, bid) = (self.survivors[i].id, self.survivors[bi].id);
                self.survivors[i].partner_id = bid;
                self.survivors[bi].partner_id = sid;
            }
        }
        for i in 0..n {
            if !self.survivors[i].alive() || !self.is_female(&self.survivors[i]) {
                continue;
            }
            if self.survivors[i].pregnant_since >= 0 {
                if self.minute - self.survivors[i].pregnant_since >= fi(self.defs.gestation_days * mpd as f64) {
                    self.birth(i);
                }
                continue;
            }
            let pa = match self.survivor_by_id(self.survivors[i].partner_id) {
                Some(pi) if self.survivors[pi].alive() => pi,
                _ => continue,
            };
            let d = &self.defs;
            let p = d.pregnancy_chance_night * curve(&d.fertility_female, self.survivors[i].age as f64) * curve(&d.fertility_male, self.survivors[pa].age as f64);
            if p > 0.0 && self.rng.randf() < p {
                self.survivors[i].pregnant_since = self.minute;
            }
        }
    }

    fn birth(&mut self, mi: usize) {
        self.survivors[mi].pregnant_since = -1;
        let safe = self.offline || self.difficulty == GENTLE;
        let r = self.rng.randf();
        let q = self.defs.birth_quality_base + if self.survivors[mi].bed_id >= 0 { self.defs.birth_quality_bed } else { 0.0 };
        let mother_dies = self.rng.randf() < curve(&self.defs.mother_death, q) && !safe;
        if r >= self.defs.birth_healthy + self.defs.birth_sick && !safe {
            self.hope = clampf(self.hope + self.defs.death_hope, 0.0, 100.0);
        } else {
            let at = self.survivors[mi].pos;
            let bi = self.add_survivor(at, CHILD);
            let (mid, pid) = (self.survivors[mi].id, self.survivors[mi].partner_id);
            let b = &mut self.survivors[bi];
            b.age = 0;
            b.age_days = 0.0;
            b.parents[0] = mid;
            if pid >= 0 {
                b.parents[1] = pid;
            }
            if r >= self.defs.birth_healthy {
                b.illness = SICK;
            }
            self.hope = clampf(self.hope + self.defs.baby_hope, 0.0, 100.0);
            self.auto_assign_beds();
        }
        if mother_dies {
            self.die(mi);
        }
    }

    fn population_floor(&mut self) {
        if self.difficulty == HARDCORE {
            return;
        }
        if self.day() % fi(self.defs.wanderer_check_days) != 0 || self.alive_count() >= fi(self.defs.pop_floor) {
            return;
        }
        if !self.furnace_lit {
            return;
        }
        for _ in 0..fi(self.defs.wanderers_per_arrival) {
            let at = self.fire_spot(self.survivors.len() as i64);
            self.add_survivor(at, WORKER);
        }
        for i in 0..self.defs.wanderer_supplies.len() {
            let (k, v) = self.defs.wanderer_supplies[i];
            self.res[k] += v;
        }
        self.auto_assign_beds();
    }

    fn roll_events(&mut self) {
        for i in 0..self.defs.waves.len() {
            let wv = self.defs.waves[i].clone();
            if fi(wv.day) != self.day() {
                continue;
            }
            let all_n = self.round_random(wv.all);
            let sick_n = self.round_random(wv.sick_all).min(all_n);
            let healthy_n = self.round_random(wv.healthy).min(all_n - sick_n);
            let day = self.day();
            self.pending.push(Choice { kind: 0, options: vec![0, 1, 2], count: all_n, sick: sick_n, healthy: healthy_n, day });
        }
    }

    fn round_random(&mut self, x: f64) -> i64 {
        fi(x.floor()) + if self.rng.randf() < x - x.floor() { 1 } else { 0 }
    }

    fn random_role(&mut self) -> u8 {
        let d = &self.defs;
        let total = d.role_worker + d.role_engineer + d.role_child;
        let x = self.rng.randf() * total;
        if x < d.role_worker {
            return WORKER;
        }
        if x < d.role_worker + d.role_engineer {
            return ENGINEER;
        }
        CHILD
    }

    /// option: 0 accept, 1 accept_healthy, 2 refuse
    pub fn resolve_choice(&mut self, index: usize, option: usize) -> bool {
        if index >= self.pending.len() {
            return false;
        }
        let c = self.pending.remove(index);
        if c.kind == 1 {
            if option == OPT_CORE {
                self.res[5] -= 1.0;
            } else if let Some(sid) = self.sacrifice_child() {
                self.kill_now(sid);
            }
            self.pressure = minf(self.pressure, self.defs.od_reset_pressure);
            return true;
        }
        if option == 2 {
            self.hope = clampf(self.hope - 6.0, 0.0, 100.0);
            self.discontent = clampf(self.discontent + 2.0, 0.0, 100.0);
        } else {
            let sick = if option == 0 { c.sick } else { 0 };
            let n = if option == 0 { c.count } else { c.healthy };
            for i in 0..n {
                let at = self.fire_spot(self.survivors.len() as i64);
                let role = self.random_role();
                let si = self.add_survivor(at, role);
                if i < sick {
                    self.survivors[si].illness = SICK;
                }
            }
            self.auto_assign_beds();
            if option == 0 {
                self.hope = clampf(self.hope + 4.0, 0.0, 100.0);
                self.discontent = clampf(self.discontent + 2.0, 0.0, 100.0);
            } else {
                self.hope = clampf(self.hope - 2.0, 0.0, 100.0);
            }
        }
        true
    }

    // ============================================================ research

    #[inline]
    pub fn has_tech(&self, t: usize) -> bool {
        self.researched.contains(&t)
    }

    pub fn tech_max_furnace(&self) -> f64 {
        let mut m = 1.0;
        for &t in &self.researched {
            if let Some(v) = self.defs.techs[t].furnace_max {
                m = maxf(m, v);
            }
        }
        m
    }
    pub fn tech_max_heater(&self) -> f64 {
        let mut m = 0.0;
        for &t in &self.researched {
            if let Some(v) = self.defs.techs[t].heater_max {
                m = maxf(m, v);
            }
        }
        m
    }
    pub fn tech_max_teams(&self) -> f64 {
        let mut m = 0.0;
        for &t in &self.researched {
            if let Some(v) = self.defs.techs[t].max_teams {
                m = maxf(m, v);
            }
        }
        m
    }
    pub fn scout_speed(&self) -> f64 {
        let mut m = 1.0;
        for &t in &self.researched {
            if let Some(v) = self.defs.techs[t].scout_speed {
                m = maxf(m, v);
            }
        }
        m
    }
    fn tech_sum_heater_coal_minus(&self) -> f64 {
        let mut s = 0.0;
        for &t in &self.researched {
            if let Some(v) = self.defs.techs[t].heater_coal_minus {
                s += v;
            }
        }
        s
    }
    fn tech_product_furnace_coal_mult(&self) -> f64 {
        let mut p = 1.0;
        for &t in &self.researched {
            if let Some(v) = self.defs.techs[t].furnace_coal_mult {
                p *= v;
            }
        }
        p
    }
    fn tech_product_cure_mult(&self) -> f64 {
        let mut p = 1.0;
        for &t in &self.researched {
            if let Some(v) = self.defs.techs[t].cure_mult {
                p *= v;
            }
        }
        p
    }

    pub fn furnace_max_level(&self) -> i64 {
        fi(self.tech_max_furnace()).min(fi(self.defs.furnace_max_level))
    }
    pub fn heater_max_level(&self) -> i64 {
        fi(self.tech_max_heater()).min(fi(self.defs.heater_max_level))
    }

    pub fn rate_mult(&self, d: usize) -> f64 {
        let mut p = 1.0;
        for &t in &self.researched {
            if let Some(m) = &self.defs.techs[t].rate_mult {
                p *= m.iter().find(|x| x.0 == d).map(|x| x.1).unwrap_or(1.0);
            }
        }
        p
    }

    pub fn insulation_of(&self, d: usize) -> f64 {
        let mut i = self.defs.buildings[d].insulation;
        for &t in &self.researched {
            if let Some(m) = &self.defs.techs[t].insulation {
                i += m.iter().find(|x| x.0 == d).map(|x| x.1).unwrap_or(0.0);
            }
        }
        i
    }

    pub fn building_locked(&self, d: usize) -> bool {
        for (ti, t) in self.defs.techs.iter().enumerate() {
            if t.unlock.contains(&d) {
                return !self.has_tech(ti);
            }
        }
        false
    }

    pub fn tier_open(&self, tier: i64) -> bool {
        if tier == 0 {
            return true;
        }
        for (ti, t) in self.defs.techs.iter().enumerate() {
            if fi(t.opens_tier) == tier {
                return self.has_tech(ti);
            }
        }
        false
    }

    pub fn can_research(&self, t: usize) -> bool {
        if self.has_tech(t) || self.research_id.is_some() {
            return false;
        }
        let td = &self.defs.techs[t];
        if !self.tier_open(fi(td.tier)) {
            return false;
        }
        for &r in &td.requires {
            if !self.has_tech(r) {
                return false;
            }
        }
        for &(k, c) in &td.cost {
            if self.res[k] < c - 0.0001 {
                return false;
            }
        }
        self.buildings.iter().any(|b| b.built && self.defs.buildings[b.def].researches)
    }

    pub fn start_research(&mut self, t: usize) -> bool {
        if !self.can_research(t) {
            return false;
        }
        for i in 0..self.defs.techs[t].cost.len() {
            let (k, c) = self.defs.techs[t].cost[i];
            self.res[k] -= c;
        }
        self.research_id = Some(t);
        self.research_done_h = 0.0;
        true
    }

    pub fn research_hours(&self, t: usize) -> f64 {
        self.defs.techs[t].hours + self.defs.difficulty_hours_add[self.difficulty]
    }

    pub fn research_speed(&self) -> f64 {
        let mut eff = 0.0f64;
        for b in &self.buildings {
            if !b.built || !self.defs.buildings[b.def].researches {
                continue;
            }
            for s in &self.survivors {
                if s.alive() && s.place_id == b.id && s.act == Act::Work {
                    eff += self.work_efficiency(s) * self.defs.efficiency_per_engineer;
                }
            }
        }
        let mut speed = 0.0f64;
        for &(cap, mult) in &self.defs.speed_bands {
            let part = minf(eff, cap);
            speed += part * mult;
            eff -= part;
            if eff <= 0.0 {
                break;
            }
        }
        speed
    }

    fn research(&mut self) {
        let t = match self.research_id {
            Some(t) => t,
            None => return,
        };
        self.research_done_h += self.research_speed() / 60.0;
        if self.research_done_h >= self.research_hours(t) - 0.00001 {
            self.researched.push(t);
            self.research_id = None;
            self.research_done_h = 0.0;
        }
    }
}

#[derive(Clone, Copy)]
enum Cause {
    Sick,
    Starve,
}

#[inline]
pub fn stage_mult(stages: &[(f64, f64)], sev: f64) -> f64 {
    let mut m = 1.0;
    for &(at, v) in stages {
        if sev >= at {
            m = v;
        }
    }
    m
}

#[inline]
fn signf(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x < 0.0 {
        -1.0
    } else {
        1.0
    }
}

fn bed_priority(s: &Survivor) -> i64 {
    let mut p = 0;
    if s.illness != NONE {
        p += 4;
    }
    if s.trait_ == TRAIT_FRAIL {
        p += 2;
    }
    if s.age >= 55 || s.role == CHILD {
        p += 1;
    }
    p
}

impl Defs {
    #[inline]
    pub fn bix_beacon(&self) -> usize {
        self.bix_cached("beacon")
    }
    pub fn bix_cached(&self, id: &str) -> usize {
        self.bix(id)
    }
}
