//! Port of sim/weather.gd (scripted FP storms and the chaotic Lorenz-63 weather).

use crate::defs::Defs;
use crate::rng::Rng;

pub const SENTINEL: i64 = 1 << 40;
pub const CALM: i64 = 0;
pub const STORM: i64 = 2;
const S: i64 = 65536;

#[derive(Clone, Debug)]
pub struct Weather {
    pub profile: usize,
    pub chaotic: bool,
    pub temp_c: f64,
    pub storm_index: i64,
    pub next_storm_start: i64,
    pub next_storm_end: i64,
    pub next_storm_temps: Vec<f64>,
    pub next_storm_end_c: f64, // NaN = none
    pub storms_survived: i64,
    pub next_storm_final_start: i64,
    pub next_storm_final_c: f64,
    pub next_storm_scenario_end: bool,
    // chaotic mode
    pub wrng: Rng,
    pub lx: i64,
    pub ly: i64,
    pub lz: i64,
    pub phase: i64,
    pub tier: i64,
    pub pressure: i64,
    pub p_hist: Vec<i64>,
    pub wind_dir: i64,
    pub wind_ms: i64,
    pub animals: i64,
    pub sky: i64,
    pub last_onset: i64,
    pub zh1: i64,
    pub zh2: i64,
    pub z_hist: Vec<i64>,
    pub last_end: i64,
    pub peak: i64,
    pub oracle_land: i64,
    pub oracle_tier: i64,
}

/// GDScript `int / int` and `%` truncate toward zero, as Rust does.
impl Weather {
    pub fn new() -> Weather {
        Weather {
            profile: 0,
            chaotic: false,
            temp_c: -20.0,
            storm_index: 0,
            next_storm_start: -1,
            next_storm_end: -1,
            next_storm_temps: Vec::new(),
            next_storm_end_c: f64::NAN,
            storms_survived: 0,
            next_storm_final_start: -1,
            next_storm_final_c: f64::NAN,
            next_storm_scenario_end: false,

            wrng: Rng::new(0), // unused (and not digested) in scripted mode
            lx: 0,
            ly: 0,
            lz: 0,
            phase: CALM,
            tier: 0,
            pressure: 10130,
            p_hist: Vec::new(),
            wind_dir: 6,
            wind_ms: 3,
            animals: 0,
            sky: 0,
            last_onset: -1,
            zh1: 0,
            zh2: 0,
            z_hist: Vec::new(),
            last_end: -SENTINEL,
            peak: 0,
            oracle_land: SENTINEL,
            oracle_tier: 0,
        }
    }

    pub fn init(&mut self, defs: &Defs, profile: usize, seed: u64) {
        self.profile = profile;
        self.chaotic = defs.weather_chaotic;
        self.temp_c = defs.profiles[profile].script[0].1;
        if self.chaotic {
            self.chaos_init(defs, seed);
        }
        self.schedule(defs, -1);
    }

    #[inline]
    fn day_len(defs: &Defs) -> i64 {
        defs.minutes_per_day as i64
    }

    fn change_minute(defs: &Defs, day: i64) -> i64 {
        day * Self::day_len(defs) + (defs.weather_change_hour as i64) * 60
    }

    fn schedule(&mut self, defs: &Defs, prev_start_day: i64) {
        if self.chaotic {
            self.phase = CALM;
            self.tier = 0;
            self.shape(defs, SENTINEL, 0);
            let from = self.last_end.max(0);
            self.oracle(defs, from);
            return;
        }
        let pr = &defs.profiles[self.profile];
        let scripted = &pr.storms;
        let day = if (self.storm_index as usize) < scripted.len() {
            scripted[self.storm_index as usize].day as i64
        } else {
            let k = self.storm_index - scripted.len() as i64;
            if prev_start_day < 0 {
                pr.rs_first_day as i64
            } else {
                prev_start_day
                    + (pr.rs_min_interval_days as i64)
                        .max(pr.rs_first_interval_days as i64 + pr.rs_interval_step_days as i64 * (k - 1).max(0))
            }
        };
        self.shape(defs, Self::change_minute(defs, day), 0);
    }

    fn shape(&mut self, defs: &Defs, start: i64, t: i64) {
        let pr = &defs.profiles[self.profile];
        let scripted = &pr.storms;
        self.next_storm_start = start;
        if (self.storm_index as usize) < scripted.len() {
            let st = &scripted[self.storm_index as usize];
            self.next_storm_temps = st.temps.clone();
            self.next_storm_end_c = st.end_c.unwrap_or(f64::NAN);
            self.next_storm_scenario_end = st.scenario_end;
        } else {
            let k = self.storm_index - scripted.len() as i64;
            let temps = &pr.rs_temps;
            let tv = temps[(k + t).clamp(0, temps.len() as i64 - 1) as usize];
            self.next_storm_temps = vec![tv; pr.rs_days as i64 as usize];
            self.next_storm_end_c = f64::NAN;
            self.next_storm_scenario_end = false;
        }
        self.next_storm_end = self.next_storm_start + self.next_storm_temps.len() as i64 * Self::day_len(defs);
        self.next_storm_final_start = -1;
        self.next_storm_final_c = f64::NAN;
        if (self.storm_index as usize) < scripted.len() {
            if let Some((sh, eh, c)) = scripted[self.storm_index as usize].final_phase {
                let dl = Self::day_len(defs);
                let end_day = self.next_storm_end / dl;
                self.next_storm_final_start = end_day * dl + (sh * 60.0) as i64;
                self.next_storm_final_c = c;
                self.next_storm_end = end_day * dl + (eh * 60.0) as i64;
            }
        }
    }

    pub fn storm_temp(&self, defs: &Defs, minute: i64) -> f64 {
        if self.next_storm_final_start >= 0 && minute >= self.next_storm_final_start {
            return self.next_storm_final_c;
        }
        let mut i = 0i64;
        if minute >= self.next_storm_start {
            i = ((minute - self.next_storm_start) / Self::day_len(defs)).clamp(0, self.next_storm_temps.len() as i64 - 1);
        }
        self.next_storm_temps[i as usize]
    }

    pub fn coldest_storm_temp(&self) -> f64 {
        let mut t = f64::INFINITY;
        for &x in &self.next_storm_temps {
            t = minf(t, x);
        }
        if self.next_storm_final_start >= 0 {
            t = minf(t, self.next_storm_final_c);
        }
        t
    }

    #[inline]
    pub fn in_storm(&self, minute: i64) -> bool {
        minute >= self.next_storm_start && minute < self.next_storm_end
    }

    pub fn forecast_active(&self, defs: &Defs, minute: i64) -> bool {
        if self.chaotic {
            return false;
        }
        let lead = defs.forecast_days as i64 * Self::day_len(defs);
        minute >= self.next_storm_start - lead && minute < self.next_storm_start
    }

    pub fn warning_active(&self, defs: &Defs, minute: i64) -> bool {
        if !self.chaotic {
            return self.forecast_active(defs, minute);
        }
        if self.in_storm(minute) {
            return false;
        }
        let (wt, ww) = (defs.chaos.warn_trend, defs.chaos.warn_wind_ms);
        self.sky == 2 || (self.pressure_trend() <= wt && self.wind_ms >= ww)
    }

    #[inline]
    pub fn outdoor_temp(&self, defs: &Defs, minute: i64) -> f64 {
        if self.in_storm(minute) {
            self.storm_temp(defs, minute)
        } else {
            self.temp_c
        }
    }

    pub fn daily_change(&mut self, defs: &Defs, rng: &mut Rng, day: i64) {
        let pr = &defs.profiles[self.profile];
        for e in &pr.script {
            if e.0 as i64 == day {
                self.temp_c = e.1;
                return;
            }
        }
        if day <= pr.script[pr.script.len() - 1].0 as i64 {
            return;
        }
        if rng.randf() < pr.random_change_chance {
            let step = if rng.randf() < 0.5 { 10.0 } else { -10.0 };
            self.temp_c = clampf(self.temp_c + step, pr.random_min_c, pr.random_max_c);
        }
    }

    pub fn storm_ended(&mut self, defs: &Defs) {
        self.storms_survived += 1;
        if !self.next_storm_end_c.is_nan() {
            self.temp_c = self.next_storm_end_c;
        }
        let start_day = self.next_storm_start / Self::day_len(defs);
        if self.chaotic {
            self.last_end = self.next_storm_end;
        }
        self.storm_index += 1;
        if self.chaotic {
            self.sky = 0;
        }
        self.schedule(defs, start_day);
    }

    // ------------------------------------------------------------ chaotic mode

    fn chaos_init(&mut self, defs: &Defs, seed: u64) {
        let c = &defs.chaos;
        self.wrng.seed(seed ^ (c.seed_xor as u64));
        self.lx = self.wrng.randi_range(-15, 15) * S;
        self.lx += self.wrng.randi_range(0, S - 1);
        self.ly = self.wrng.randi_range(-15, 15) * S;
        self.ly += self.wrng.randi_range(0, S - 1);
        self.lz = self.wrng.randi_range(10, 40) * S;
        self.lz += self.wrng.randi_range(0, S - 1);
        for _ in 0..c.spinup_steps {
            self.lorenz_step(defs);
        }
        self.phase = CALM;
        self.p_hist.clear();
        self.zh1 = self.lz;
        self.zh2 = self.lz;
        self.z_hist.clear();
        self.last_end = -SENTINEL;
        self.readings(defs);
    }

    #[inline]
    fn lorenz_step(&mut self, defs: &Defs) {
        let c = &defs.chaos;
        let dt = c.dt_div;
        let dx = c.sigma * (self.ly - self.lx) / dt;
        let dy = (self.lx * (c.rho * S - self.lz) / S - self.ly) / dt;
        let dz = (c.beta_den * (self.lx * self.ly / S) - c.beta_num * self.lz) / (c.beta_den * dt);
        self.lx += dx;
        self.ly += dy;
        self.lz += dz;
    }

    /// See sim/weather.gd `_lands`.
    fn lands(&self, defs: &Defs, idx: i64, minute: i64, z: i64, after_end: i64) -> bool {
        let c = &defs.chaos;
        let dl = Self::day_len(defs);
        if (idx as usize) < defs.profiles[self.profile].storms.len() {
            if minute >= c.sc_force_day * dl {
                return true;
            }
            return minute >= c.sc_open_day * dl && z > c.sc_peak_c * S / 100;
        }
        z > c.peak_c * S / 100 && minute >= c.first_day * dl && minute >= after_end + c.min_gap_days * dl
    }

    fn tier_of(defs: &Defs, z: i64) -> i64 {
        let c = &defs.chaos;
        if z < c.tier_shallow_below_c * S / 100 {
            -1
        } else if z >= c.tier_deep_from_c * S / 100 {
            1
        } else {
            0
        }
    }

    pub fn hourly(&mut self, defs: &Defs, minute: i64) {
        for _ in 0..defs.chaos.steps_per_hour {
            self.lorenz_step(defs);
        }
        let landed = self.phase == CALM
            && self.zh1 > self.zh2
            && self.zh1 >= self.lz
            && self.lands(defs, self.storm_index, minute, self.zh1, self.last_end);
        if landed {
            self.peak = self.zh1;
            self.tier = Self::tier_of(defs, self.zh1);
            let t = self.tier;
            self.shape(defs, minute, t);
            self.phase = STORM;
            self.last_onset = minute;
            self.animals = 2;
            self.sky = 3;
        }
        self.zh2 = self.zh1;
        self.zh1 = self.lz;
        if landed {
            self.oracle(defs, minute);
        }
        self.readings(defs);
        if minute % Self::day_len(defs) == (defs.weather_change_hour as i64) * 60 {
            self.daily_signs(defs);
        }
    }

    /// The true next landing after `from` (see sim/weather.gd `_oracle`).
    pub fn oracle(&mut self, defs: &Defs, from: i64) {
        let c = &defs.chaos;
        let (mut sx, mut sy, mut sz) = (self.lx, self.ly, self.lz);
        let (mut h1, mut h2) = (self.zh1, self.zh2);
        let storm = self.phase == STORM;
        let after_end = if storm { self.next_storm_end } else { self.last_end };
        let idx = if storm { self.storm_index + 1 } else { self.storm_index };
        self.oracle_land = SENTINEL;
        self.oracle_tier = 0;
        let dl = Self::day_len(defs);
        let mut m = from - from % 60;
        let stop = from + c.oracle_horizon_days * dl;
        let rho = c.rho * S;
        while m < stop {
            m += 60;
            for _ in 0..c.steps_per_hour {
                let dx = c.sigma * (sy - sx) / c.dt_div;
                let dy = (sx * (rho - sz) / S - sy) / c.dt_div;
                let dz = (c.beta_den * (sx * sy / S) - c.beta_num * sz) / (c.beta_den * c.dt_div);
                sx += dx;
                sy += dy;
                sz += dz;
            }
            if h1 > h2 && h1 >= sz && (!storm || m > after_end) && self.lands(defs, idx, m, h1, after_end) {
                self.oracle_land = m;
                self.oracle_tier = Self::tier_of(defs, h1);
                return;
            }
            h2 = h1;
            h1 = sz;
        }
    }

    fn readings(&mut self, defs: &Defs) {
        let c = &defs.chaos;
        self.pressure = c.p_base - (self.lz - c.p_z_mid * S) * c.p_z_gain / S + self.wrng.randi_range(-c.p_noise, c.p_noise);
        self.p_hist.push(self.pressure);
        if self.p_hist.len() > 25 {
            self.p_hist.remove(0);
        }
        self.z_hist.push(self.lz);
        if self.z_hist.len() > 24 {
            self.z_hist.remove(0);
        }
        let noise = self.wrng.randi_range(-c.w_noise, c.w_noise);
        if self.phase == STORM {
            self.wind_dir = if self.ly >= 0 { 0 } else { 1 };
            self.wind_ms = c.w_storm_base + self.ly.abs() / (S * c.w_y_div) + noise;
        } else {
            self.wind_dir = if self.lx >= 0 {
                if self.ly >= 0 {
                    6
                } else {
                    5
                }
            } else if self.ly > 8 * S {
                7
            } else if self.ly < -8 * S {
                1
            } else {
                0
            };
            self.wind_ms = c.w_base + self.ly.abs() / (S * c.w_y_div) + noise;
        }
        self.wind_ms = self.wind_ms.max(0);
    }

    fn daily_signs(&mut self, defs: &Defs) {
        let c = &defs.chaos;
        let mut zmax = self.lz;
        for &v in &self.z_hist {
            zmax = zmax.max(v);
        }
        let stage = if self.phase == STORM || zmax >= c.z2_c * S / 100 {
            2
        } else if zmax >= c.z1_c * S / 100 {
            1
        } else {
            0
        };
        let r = self.wrng.randi_range(0, 99);
        let r2 = self.wrng.randi_range(0, 99);
        let ap = c.animal_noise_pct;
        let n = if r < ap {
            -1
        } else if r >= 100 - ap {
            1
        } else {
            0
        };
        self.animals = (stage + n).clamp(0, 2);
        self.sky = if self.phase == STORM {
            3
        } else if stage == 0 {
            if r2 < c.halo_calm_pct {
                1
            } else {
                0
            }
        } else if stage == 1 {
            if r2 < c.halo_s1_pct {
                1
            } else {
                0
            }
        } else if r2 < c.grey_s2_pct {
            2
        } else {
            1
        };
    }

    pub fn pressure_trend(&self) -> i64 {
        if self.p_hist.is_empty() {
            return 0;
        }
        self.p_hist[self.p_hist.len() - 1] - self.p_hist[0]
    }
}

/// Godot minf/maxf/clampf (Math::min etc. with `a < b ? a : b` semantics).
#[inline]
pub fn minf(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}
#[inline]
pub fn maxf(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}
#[inline]
pub fn clampf(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}
