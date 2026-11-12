//! Python bindings: `lastembers_sim.Env` (one env) and `lastembers_sim.VecEnv`
//! (N envs stepped in parallel with the GIL released). Values are exactly what
//! tools/rl_server.gd returns after its JSON round trip (obs as float32).

use lastembers_core::defs::Defs;
use lastembers_core::env::{json_f64, obs_f32, Env as CoreEnv, ResetOpts};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::path::PathBuf;

fn load_defs(data_dir: Option<String>) -> PyResult<Defs> {
    let dir = match data_dir.or_else(|| std::env::var("LASTEMBERS_DATA").ok()) {
        Some(d) => PathBuf::from(d),
        None => return Err(PyValueError::new_err("pass data_dir (the game's data/ folder) or set LASTEMBERS_DATA")),
    };
    std::panic::catch_unwind(|| Defs::load_dir(&dir)).map_err(|_| PyValueError::new_err(format!("cannot load game data from {}", dir.display())))
}

#[allow(clippy::too_many_arguments)]
fn opts(
    seed: i64,
    perturb: bool,
    warm: f64,
    no_floor: bool,
    difficulty: &str,
    weather: &str,
    precursor_obs: bool,
    oracle_obs: bool,
    max_days: i64,
    weather_noise: &str,
) -> ResetOpts {
    ResetOpts {
        seed,
        perturb,
        warm,
        no_floor,
        difficulty: difficulty.to_string(),
        weather: weather.to_string(),
        precursor_obs,
        oracle_obs,
        max_days,
        weather_noise: weather_noise.to_string(),
    }
}

fn info<'py>(py: Python<'py>, day: i64, alive: i64) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("day", day)?;
    d.set_item("alive", alive)?;
    Ok(d)
}

#[pyclass(unsendable)]
struct Env {
    e: CoreEnv,
    obs: Vec<f64>,
    mask: Vec<bool>,
}

#[pymethods]
impl Env {
    #[new]
    #[pyo3(signature = (data_dir=None))]
    fn new(data_dir: Option<String>) -> PyResult<Self> {
        Ok(Env { e: CoreEnv::new(load_defs(data_dir)?), obs: Vec::new(), mask: Vec::new() })
    }

    #[getter]
    fn action_names(&self) -> Vec<String> {
        self.e.names.clone()
    }
    #[getter]
    fn n_actions(&self) -> usize {
        self.e.n_actions()
    }
    #[getter]
    fn obs_size(&self) -> usize {
        self.e.obs().len()
    }

    /// reset(seed, perturb=False, warm=0.0, no_floor=True, difficulty="STANDARD", weather="scripted",
    ///       precursor_obs=False, oracle_obs=False, max_days=0, weather_noise="murky") -> (obs, mask, info)
    #[pyo3(signature = (seed=1, perturb=false, warm=0.0, no_floor=true, difficulty="STANDARD", weather="scripted", precursor_obs=false, oracle_obs=false, max_days=0, weather_noise="murky"))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: i64,
        perturb: bool,
        warm: f64,
        no_floor: bool,
        difficulty: &str,
        weather: &str,
        precursor_obs: bool,
        oracle_obs: bool,
        max_days: i64,
        weather_noise: &str,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyDict>)> {
        let o = self.e.reset(&opts(seed, perturb, warm, no_floor, difficulty, weather, precursor_obs, oracle_obs, max_days, weather_noise));
        let obs: Vec<f32> = o.obs.iter().map(|x| obs_f32(*x)).collect();
        Ok((obs.into_pyarray(py), o.mask.into_pyarray(py), info(py, o.day, o.alive)?))
    }

    /// step(action) -> (obs, mask, reward, done, info)
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: i64,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>, f64, bool, Bound<'py, PyDict>)> {
        let (r, done) = self.e.step_raw(action);
        self.e.obs_into(&mut self.obs);
        self.e.mask_into(&mut self.mask);
        let obs: Vec<f32> = self.obs.iter().map(|x| obs_f32(*x)).collect();
        Ok((obs.into_pyarray(py), self.mask.clone().into_pyarray(py), json_f64(r), done, info(py, self.e.w.day(), self.e.w.alive_count())?))
    }

    /// Independent deep copy of this env (full sim state incl. RNG); stepping the copy never affects self.
    fn clone(&self) -> Self {
        Env { e: self.e.clone(), obs: self.obs.clone(), mask: self.mask.clone() }
    }

    /// The raw f64 observation GDScript computes (before JSON rounding).
    fn raw_obs(&self) -> Vec<f64> {
        self.e.obs()
    }
}

struct Slot {
    e: CoreEnv,
    obs: Vec<f64>,
    mask: Vec<bool>,
    reward: f64,
    done: bool,
}

/// N independent envs; step() runs them in parallel without the GIL.
#[pyclass]
struct VecEnv {
    slots: Vec<Slot>,
    pool: rayon::ThreadPool,
    n_act: usize,
    obs_dim: usize,
    precursor_obs: bool,
    oracle_obs: bool,
}

impl VecEnv {
    fn gather<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
        let n = self.slots.len();
        let mut o = Vec::with_capacity(n * self.obs_dim);
        let mut m = Vec::with_capacity(n * self.n_act);
        for s in &self.slots {
            o.extend(s.obs.iter().map(|x| obs_f32(*x)));
            m.extend_from_slice(&s.mask);
        }
        let o = o.into_pyarray(py).reshape([n, self.obs_dim])?;
        let m = m.into_pyarray(py).reshape([n, self.n_act])?;
        Ok((o, m))
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n, data_dir=None, threads=0, precursor_obs=false, oracle_obs=false))]
    fn new(n: usize, data_dir: Option<String>, threads: usize, precursor_obs: bool, oracle_obs: bool) -> PyResult<Self> {
        let defs = load_defs(data_dir)?;
        let mut proto = CoreEnv::new(defs);
        proto.precursor_obs = precursor_obs;
        proto.oracle_obs = oracle_obs;
        let n_act = proto.n_actions();
        let obs_dim = proto.obs().len();
        let slots = (0..n)
            .map(|_| {
                let e = proto.clone();
                let obs = e.obs();
                let mask = e.mask();
                Slot { e, obs, mask, reward: 0.0, done: false }
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VecEnv { slots, pool, n_act, obs_dim, precursor_obs, oracle_obs })
    }

    #[getter]
    fn action_names(&self) -> Vec<String> {
        self.slots[0].e.names.clone()
    }
    #[getter]
    fn n_actions(&self) -> usize {
        self.n_act
    }
    #[getter]
    fn obs_size(&self) -> usize {
        self.obs_dim
    }
    fn __len__(&self) -> usize {
        self.slots.len()
    }

    /// reset_at(i, seed, ...) -> (obs[i], mask[i]); resets one env.
    #[pyo3(signature = (i, seed, perturb=false, warm=0.0, no_floor=true, difficulty="STANDARD", weather="scripted", max_days=0, weather_noise="murky"))]
    fn reset_at<'py>(
        &mut self,
        py: Python<'py>,
        i: usize,
        seed: i64,
        perturb: bool,
        warm: f64,
        no_floor: bool,
        difficulty: &str,
        weather: &str,
        max_days: i64,
        weather_noise: &str,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>)> {
        let (po, oo) = (self.precursor_obs, self.oracle_obs);
        let s = self.slots.get_mut(i).ok_or_else(|| PyValueError::new_err("index out of range"))?;
        let o = s.e.reset(&opts(seed, perturb, warm, no_floor, difficulty, weather, po, oo, max_days, weather_noise));
        s.obs = o.obs;
        s.mask = o.mask;
        s.done = false;
        let obs: Vec<f32> = s.obs.iter().map(|x| obs_f32(*x)).collect();
        Ok((obs.into_pyarray(py), s.mask.clone().into_pyarray(py)))
    }

    /// reset(seeds, ...) -> (obs[N, D], mask[N, A]); resets every env with its own seed.
    #[pyo3(signature = (seeds, perturb=false, warm=0.0, no_floor=true, difficulty="STANDARD", weather="scripted", max_days=0, weather_noise="murky"))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seeds: Vec<i64>,
        perturb: bool,
        warm: f64,
        no_floor: bool,
        difficulty: &str,
        weather: &str,
        max_days: i64,
        weather_noise: &str,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
        if seeds.len() != self.slots.len() {
            return Err(PyValueError::new_err("need one seed per env"));
        }
        let o = opts(0, perturb, warm, no_floor, difficulty, weather, self.precursor_obs, self.oracle_obs, max_days, weather_noise);
        let slots = &mut self.slots;
        let pool = &self.pool;
        py.detach(|| {
            pool.install(|| {
                slots.par_iter_mut().zip(seeds.par_iter()).for_each(|(s, &seed)| {
                    let mut oo = o.clone();
                    oo.seed = seed;
                    let r = s.e.reset(&oo);
                    s.obs = r.obs;
                    s.mask = r.mask;
                    s.done = false;
                })
            })
        });
        self.gather(py)
    }

    /// step(actions[N]) -> (obs[N, D], mask[N, A], reward[N], done[N], day[N], alive[N]).
    /// Envs are not auto-reset: call reset_at for the done ones.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<i64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let acts: Vec<i64> = actions.as_slice()?.to_vec();
        if acts.len() != self.slots.len() {
            return Err(PyValueError::new_err("need one action per env"));
        }
        let slots = &mut self.slots;
        let pool = &self.pool;
        py.detach(|| {
            pool.install(|| {
                slots.par_iter_mut().zip(acts.par_iter()).for_each(|(s, &a)| {
                    let (r, d) = s.e.step_raw(a);
                    s.reward = r;
                    s.done = d;
                    s.e.obs_into(&mut s.obs);
                    s.e.mask_into(&mut s.mask);
                })
            })
        });
        let (o, m) = self.gather(py)?;
        let r: Vec<f64> = self.slots.iter().map(|s| json_f64(s.reward)).collect();
        let d: Vec<bool> = self.slots.iter().map(|s| s.done).collect();
        let day: Vec<i64> = self.slots.iter().map(|s| s.e.w.day()).collect();
        let alive: Vec<i64> = self.slots.iter().map(|s| s.e.w.alive_count()).collect();
        Ok((o, m, r.into_pyarray(py), d.into_pyarray(py), day.into_pyarray(py), alive.into_pyarray(py)))
    }
}

#[pymodule]
fn lastembers_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<VecEnv>()?;
    Ok(())
}
