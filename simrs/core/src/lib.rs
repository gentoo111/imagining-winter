//! Last Embers simulation core: a bit-exact Rust port of the GDScript sim
//! (sim/*.gd), the scripted bot (tools/bot_player.gd) and the RL environment
//! (tools/rl_server.gd). Pure Rust, no Python types, so it can also back a
//! GDExtension later. Parity harness: simrs/parity/.

pub mod bot;
pub mod defs;
pub mod env;
pub mod rng;
pub mod weather;
pub mod world;

pub use env::{Env, ResetOpts, StepOut};
