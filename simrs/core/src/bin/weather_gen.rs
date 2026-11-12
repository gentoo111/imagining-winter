//! Weather-only data for the forecasting experiments (E1). Same stepping order as
//! weather_trace (and the game): a storm ending mid-hour ends before the next hour.
//!   weather_gen <data dir> <out file> <seed_from> <n_seeds> <days> <profile> <noise> [wind_noise=<p>]
//! Optional wind_noise=<p> (offline experiments only, S1-12; the game is unchanged): each hour, with
//! probability p the *exported* wind_dir is moved one step on the 8-point compass ring (+1 or -1, p/2 each),
//! drawn from a separate per-seed Rng so the weather itself evolves exactly as without it.
//! Without the argument (or p = 0) the output is byte-identical to before.
//! Output: little-endian i32, [n_seeds][days*24][F] with F = 15 fields per game hour:
//!   minute, lx, ly, lz, pressure, trend, wind_dir, wind_ms, animals, sky,
//!   in_storm, landed (1 at the landing hour), tier (at landing, else 0), storm_index, last_end_hour
use lastembers_core::defs::Defs;
use lastembers_core::rng::Rng;
use lastembers_core::weather::{Weather, STORM};
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut defs = Defs::load_dir(std::path::Path::new(&a[1]));
    let out = &a[2];
    let from: u64 = a[3].parse().unwrap();
    let n: u64 = a[4].parse().unwrap();
    let days: i64 = a[5].parse().unwrap();
    let profile = &a[6];
    let noise = &a[7];
    let wind_noise: f64 = match a.get(8) {
        None => 0.0,
        Some(s) => s.strip_prefix("wind_noise=").expect("8th argument must be wind_noise=<p>").parse().unwrap(),
    };
    assert!((0.0..=1.0).contains(&wind_noise), "wind_noise must be in [0, 1]");
    defs.weather_chaotic = true;
    defs.chaos.set_noise(noise);
    let prof = defs.profiles.iter().position(|p| &p.id == profile).unwrap();
    let mut buf: Vec<u8> = Vec::with_capacity((n * days as u64 * 24 * 15 * 4) as usize);
    for sd in from..from + n {
        let mut w = Weather::new();
        w.init(&defs, prof, sd);
        let mut last_end_h: i64 = -100000;
        let mut nrng = if wind_noise > 0.0 { Some(Rng::new(sd ^ 0x9E37_79B9_7F4A_7C15)) } else { None };
        for h in 1..=days * 24 {
            let m = h * 60;
            if w.phase == STORM && w.next_storm_end < m && w.next_storm_end > m - 60 {
                last_end_h = w.next_storm_end / 60;
                w.storm_ended(&defs);
            }
            w.hourly(&defs, m);
            let landed = m == w.next_storm_start;
            let tier = if landed { w.tier } else { 0 };
            if m == w.next_storm_end {
                last_end_h = h;
                w.storm_ended(&defs);
            }
            let in_storm = if w.phase == STORM { 1 } else { 0 };
            let mut wind_dir = w.wind_dir;
            if let Some(r) = nrng.as_mut() {
                let u = r.next_u32() as f64 / 4294967296.0;
                if u < wind_noise / 2.0 {
                    wind_dir = (wind_dir + 1) % 8;
                } else if u < wind_noise {
                    wind_dir = (wind_dir + 7) % 8;
                }
            }
            for v in [
                m, w.lx, w.ly, w.lz, w.pressure, w.pressure_trend(), wind_dir, w.wind_ms, w.animals, w.sky,
                in_storm, landed as i64, tier, w.storm_index, last_end_h,
            ] {
                buf.extend_from_slice(&(v as i32).to_le_bytes());
            }
        }
    }
    std::fs::File::create(out).unwrap().write_all(&buf).unwrap();
}
