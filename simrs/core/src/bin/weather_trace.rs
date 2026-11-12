//! Weather-only parity for the chaotic weather; same output as tools/weather_trace.gd:
//!   weather_trace <data dir> [seeds] [days] [profile] [noise]
use lastembers_core::defs::Defs;
use lastembers_core::weather::{Weather, STORM};
use sha2::{Digest, Sha256};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut defs = Defs::load_dir(std::path::Path::new(&args[1]));
    let seeds: u64 = args.get(2).map(|x| x.parse().unwrap()).unwrap_or(200);
    let days: i64 = args.get(3).map(|x| x.parse().unwrap()).unwrap_or(150);
    let profile = args.get(4).cloned().unwrap_or("endurance".into());
    let noise = args.get(5).cloned().unwrap_or("murky".into());
    defs.weather_chaotic = true;
    defs.chaos.set_noise(&noise);
    let prof = defs.profiles.iter().position(|p| p.id == profile).unwrap();
    for sd in 1..=seeds {
        let mut w = Weather::new();
        w.init(&defs, prof, sd);
        let mut s = String::new();
        let mut storms = 0;
        for h in 1..=days * 24 {
            let m = h * 60;
            if w.phase == STORM && w.next_storm_end < m && w.next_storm_end > m - 60 {
                w.storm_ended(&defs); // game order: a storm ending mid-hour ends before the next hour
            }
            w.hourly(&defs, m);
            if m == w.next_storm_start {
                storms += 1;
            }
            if m == w.next_storm_end {
                w.storm_ended(&defs);
            }
            s.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{};",
                m, w.lx, w.ly, w.lz, w.pressure, w.pressure_trend(), w.wind_dir, w.wind_ms, w.animals, w.sky,
                w.next_storm_start, w.next_storm_end, w.oracle_land, w.oracle_tier, w.tier, w.storm_index
            ));
        }
        if std::env::var("WT_DUMP").ok().as_deref() == Some(&sd.to_string()) {
            for x in s.split_inclusive(';') {
                println!("{x}");
            }
        }
        let hx: String = Sha256::digest(s.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        println!("{sd} {storms} {hx}");
    }
}
