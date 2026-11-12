//! Godot 4.7.2 RandomNumberGenerator (core/math/random_pcg.*, thirdparty/misc/pcg.*).
//! real_t is f32 in standard builds, so randf/randf_range work in f32.

pub const DEFAULT_INC: u64 = 1442695040888963407;

#[derive(Clone, Debug)]
pub struct Rng {
    pub state: u64,
    pub inc: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut r = Rng { state: 0, inc: 0 };
        r.seed(seed);
        r
    }

    /// `rng.seed = x`: pcg32_srandom_r(seed, DEFAULT_INC).
    pub fn seed(&mut self, seed: u64) {
        self.state = 0;
        self.inc = (DEFAULT_INC << 1) | 1;
        self.next_u32();
        self.state = self.state.wrapping_add(seed);
        self.next_u32();
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc | 1);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    #[inline]
    fn bounded(&mut self, bound: u32) -> u32 {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % bound;
            }
        }
    }

    /// randi()
    pub fn randi(&mut self) -> u32 {
        self.next_u32()
    }

    /// randf(): f32 in [0,1], widened by GDScript to f64.
    #[inline]
    pub fn randf32(&mut self) -> f32 {
        let proto = self.next_u32();
        if proto == 0 {
            return 0.0;
        }
        let sig = (self.next_u32() | 0x8000_0001) as f32;
        ldexpf(sig, -32 - proto.leading_zeros() as i32)
    }

    #[inline]
    pub fn randf(&mut self) -> f64 {
        self.randf32() as f64
    }

    /// randf_range(from, to): arguments are real_t (f32).
    #[inline]
    pub fn randf_range(&mut self, from: f64, to: f64) -> f64 {
        let (a, b) = (from as f32, to as f32);
        (self.randf32() * (b - a) + a) as f64
    }

    /// randi_range(from, to): int arguments (C int, 32 bit).
    pub fn randi_range(&mut self, from: i64, to: i64) -> i64 {
        let (from, to) = (from as i32 as i64, to as i32 as i64);
        if from == to {
            return from;
        }
        let min = from.min(to);
        let max = from.max(to);
        let diff = (max - min) as u32;
        if diff == u32::MAX {
            return (self.next_u32() as i64 + min) as i32 as i64;
        }
        (self.bounded(diff.wrapping_add(1)) as i64 + min) as i32 as i64
    }
}

/// Exact ldexp for the normal range used here (exponent -32..-63, value < 2^32).
#[inline]
fn ldexpf(x: f32, e: i32) -> f32 {
    // x*2^e is exact (no subnormals for e >= -63 with x >= 2^31), so two exact power-of-two multiplies.
    let half = e / 2;
    x * f32::from_bits(((127 + half) as u32) << 23) * f32::from_bits(((127 + e - half) as u32) << 23)
}
