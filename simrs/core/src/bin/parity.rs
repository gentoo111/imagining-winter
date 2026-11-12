//! Replays tools/rl_trace.gd traces and checks every step bit for bit.
//! usage: parity <data_dir> <trace.jsonl>...   (exit 1 on the first mismatch per file)

use lastembers_core::defs::Defs;
use lastembers_core::env::{json_f64, obs_f32, Env, ResetOpts, StepOut};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::BufRead;

fn sha16(b: &[u8]) -> String {
    let h = Sha256::digest(b);
    h.iter().take(8).map(|x| format!("{x:02x}")).collect()
}

fn digest_bytes(ints: &[i64], fl: &[f64]) -> Vec<u8> {
    let mut v = Vec::with_capacity(ints.len() * 8 + fl.len() * 8);
    for i in ints {
        v.extend_from_slice(&i.to_le_bytes());
    }
    for f in fl {
        v.extend_from_slice(&f.to_le_bytes());
    }
    v
}

fn check(env: &mut Env, path: &str) -> Result<(usize, String), String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut lines = std::io::BufReader::new(f).lines();
    let head: Value = serde_json::from_str(&lines.next().ok_or("empty")?.unwrap()).unwrap();
    let r = &head["reset"];
    let opts = ResetOpts {
        seed: r["seed"].as_i64().unwrap(),
        perturb: r["perturb"].as_bool().unwrap(),
        warm: r["warm"].to_string().parse().unwrap(),
        no_floor: r["no_floor"].as_bool().unwrap(),
        difficulty: r["difficulty"].as_str().unwrap().to_string(),
        weather: r.get("weather").and_then(|x| x.as_str()).unwrap_or("scripted").to_string(),
        precursor_obs: r.get("precursor_obs").and_then(|x| x.as_bool()).unwrap_or(false),
        oracle_obs: r.get("oracle_obs").and_then(|x| x.as_bool()).unwrap_or(false),
        max_days: r.get("max_days").and_then(|x| x.as_i64()).unwrap_or(0),
        weather_noise: r.get("weather_noise").and_then(|x| x.as_str()).unwrap_or("").to_string(),
    };
    if head["n_actions"].as_u64().unwrap() as usize != env.n_actions() {
        return Err(format!("action count {} vs {}", head["n_actions"], env.n_actions()));
    }
    let mut out: StepOut = env.reset(&opts);
    let mut n = 0usize;
    for (k, line) in lines.enumerate() {
        let rec: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let a = rec["a"].as_i64().unwrap();
        if a >= 0 {
            out = env.step(a);
        }
        let mut errs = Vec::new();
        let l = Env::line(&out);
        let obs_b: Vec<u8> = out.obs.iter().flat_map(|x| x.to_le_bytes()).collect();
        let (ints, fl) = env.digest();
        let st = sha16(&digest_bytes(&ints, &fl));
        let mask_s: String = out.mask.iter().map(|m| if *m { '1' } else { '0' }).collect();
        if sha16(&obs_b) != rec["o"].as_str().unwrap() {
            errs.push("obs".to_string());
        }
        if st != rec["s"].as_str().unwrap() {
            errs.push("state".to_string());
        }
        if mask_s != rec["m"].as_str().unwrap() {
            errs.push(format!("mask\n  rust {mask_s}\n  gd   {}", rec["m"]));
        }
        let rhex: String = out.reward.to_le_bytes().iter().map(|x| format!("{x:02x}")).collect();
        if rhex != rec["r"].as_str().unwrap() {
            errs.push(format!("reward {} vs {}", rhex, rec["r"]));
        }
        if out.done != rec["d"].as_bool().unwrap() || out.day != rec["day"].as_i64().unwrap() || out.alive != rec["alive"].as_i64().unwrap() {
            errs.push(format!("done/day/alive {} {} {} vs {} {} {}", out.done, out.day, out.alive, rec["d"], rec["day"], rec["alive"]));
        }
        if sha16(l.as_bytes()) != rec["l"].as_str().unwrap() {
            errs.push("line".to_string());
            if let Some(gl) = rec.get("line") {
                errs.push(format!("  rust {l}\n  gd   {}", gl.as_str().unwrap()));
            }
        }
        // what Python sees: json.loads -> float32 (fast path must agree with the slow one)
        for x in &out.obs {
            if obs_f32(*x).to_bits() != (json_f64(*x) as f32).to_bits() {
                errs.push(format!("obs_f32 fast path mismatch for {x:e}"));
            }
        }
        if !errs.is_empty() {
            let mut msg = format!("{path}: step {k} (action {a}) mismatch: {}", errs.join("; "));
            if let (Some(gi), Some(gf)) = (rec.get("ints"), rec.get("floats")) {
                let gi: Vec<i64> = gi.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
                let gf: Vec<f64> = gf
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| {
                        let h = x.as_str().unwrap();
                        let mut b = [0u8; 8];
                        for i in 0..8 {
                            b[i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap();
                        }
                        f64::from_le_bytes(b)
                    })
                    .collect();
                msg += &format!("\n  ints len {} vs {}; floats len {} vs {}", ints.len(), gi.len(), fl.len(), gf.len());
                let mut shown = 0;
                for i in 0..ints.len().min(gi.len()) {
                    if ints[i] != gi[i] && shown < 10 {
                        msg += &format!("\n  int[{i}] rust {} gd {}", ints[i], gi[i]);
                        shown += 1;
                    }
                }
                shown = 0;
                for i in 0..fl.len().min(gf.len()) {
                    if fl[i].to_bits() != gf[i].to_bits() && shown < 10 {
                        msg += &format!("\n  float[{i}] rust {:e} ({:016x}) gd {:e} ({:016x})", fl[i], fl[i].to_bits(), gf[i], gf[i].to_bits());
                        shown += 1;
                    }
                }
            }
            return Err(msg);
        }
        n += 1;
    }
    Ok((n, format!("day {} alive {}", out.day, out.alive)))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let defs = Defs::load_dir(std::path::Path::new(&args[1]));
    let mut env = Env::new(defs);
    let (mut ok, mut bad, mut steps) = (0, 0, 0usize);
    for p in &args[2..] {
        match check(&mut env, p) {
            Ok((n, s)) => {
                ok += 1;
                steps += n;
                if std::env::var("VERBOSE").is_ok() {
                    println!("OK {p} {n} steps {s}");
                }
            }
            Err(e) => {
                bad += 1;
                println!("FAIL {e}");
            }
        }
    }
    println!("parity: {ok} ok, {bad} failed, {steps} steps matched");
    std::process::exit(if bad > 0 { 1 } else { 0 });
}
