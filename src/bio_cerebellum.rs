//! bio_cerebellum (motor #3): a forward model learned by climbing-fiber error — sparse granule expansion
//! (Marr–Albus, Golgi-normalized) + one climbing-fibre error per Purkinje cell (local, no backprop). The
//! expansion makes the nonlinear model learnable; the climbing fibre is the teacher. From bio_cerebellum.py.

use crate::rng::Rng;
use crate::vec;
use std::f64::consts::PI;

pub struct Cerebellum {
    n_granule: usize,
    n_in: usize,
    k: usize,
    lr: f64,
    wg: Vec<f64>, // n_granule × n_in
    bg: Vec<f64>,
    w: Vec<f64>, // parallel-fiber → Purkinje
}

impl Cerebellum {
    pub fn new(seed: u64) -> Self {
        let (n_granule, n_in, k) = (400usize, 3usize, 40usize);
        let mut g = Rng::new(seed);
        let wg = g.randn_vec(n_granule * n_in, 1.0);
        let bg = g.randn_vec(n_granule, 0.4);
        Cerebellum { n_granule, n_in, k, lr: 0.3, wg, bg, w: vec![0.0; n_granule] }
    }
    fn granule(&self, x: &[f64]) -> Vec<f64> {
        let mut pre = vec::mv(&self.wg, x, self.n_granule, self.n_in);
        for i in 0..self.n_granule {
            pre[i] += self.bg[i];
        }
        let idx = vec::topk_indices(&pre, self.k);
        let mut g = vec![0.0; self.n_granule];
        for &i in &idx {
            g[i] = pre[i].max(0.0);
        }
        let norm = vec::norm(&g) + 1e-8; // Golgi-cell normalization (keeps the LMS rule stable)
        for gi in g.iter_mut() {
            *gi /= norm;
        }
        g
    }
    pub fn predict(&self, x: &[f64]) -> f64 {
        vec::dot(&self.w, &self.granule(x))
    }
    pub fn teach(&mut self, x: &[f64], y: f64, cf_gain: f64) {
        let g = self.granule(x);
        let e = vec::dot(&self.w, &g) - y; // climbing-fiber error
        for i in 0..self.n_granule {
            self.w[i] -= self.lr * cf_gain * e * g[i];
        }
    }
}

/// Control: the same local delta rule on the raw input (no granule expansion).
struct LinearForward {
    w: Vec<f64>,
    lr: f64,
}
impl LinearForward {
    fn new() -> Self {
        LinearForward { w: vec![0.0; 3], lr: 0.05 }
    }
    fn predict(&self, x: &[f64]) -> f64 {
        vec::dot(&self.w, x)
    }
    fn teach(&mut self, x: &[f64], y: f64) {
        let e = vec::dot(&self.w, x) - y;
        for i in 0..3 {
            self.w[i] -= self.lr * e * x[i];
        }
    }
}

/// A motor state (command u, context φ) and its nonlinear, conjunctive sensory consequence.
fn sample(rng: &mut Rng) -> ([f64; 3], f64) {
    let u = rng.uniform() * 2.0 - 1.0;
    let phi = rng.uniform() * 2.0 * PI;
    ([u, phi.sin(), phi.cos()], u * (2.0 * phi).sin() + 0.5 * phi.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cb_mse(cb: &Cerebellum, rng: &mut Rng) -> f64 {
        let mut se = 0.0;
        for _ in 0..400 {
            let (x, y) = sample(rng);
            se += (cb.predict(&x) - y).powi(2);
        }
        se / 400.0
    }

    #[test]
    fn forward_model_from_climbing_fiber_error() {
        let trials = 6000;
        let mut gen = Rng::new(1);

        let mut cb = Cerebellum::new(0);
        for _ in 0..trials {
            let (x, y) = sample(&mut gen);
            cb.teach(&x, y, 1.0);
        }
        let train_mse = cb_mse(&cb, &mut Rng::new(7));
        assert!(train_mse < 0.05, "learns the forward model ({})", train_mse);
        assert!(cb_mse(&cb, &mut Rng::new(99)) < 0.06, "generalizes to held-out states");

        // no granule expansion → the linear control stalls on the conjunctive term
        let mut lin = LinearForward::new();
        for _ in 0..trials {
            let (x, y) = sample(&mut gen);
            lin.teach(&x, y);
        }
        let mut se = 0.0;
        let mut tg = Rng::new(7);
        for _ in 0..400 {
            let (x, y) = sample(&mut tg);
            se += (lin.predict(&x) - y).powi(2);
        }
        assert!(se / 400.0 > 4.0 * train_mse, "granule expansion is necessary");

        // silence the climbing fibre → no learning (it is the teacher, not bare Hebbian)
        let mut cb0 = Cerebellum::new(0);
        for _ in 0..trials {
            let (x, y) = sample(&mut gen);
            cb0.teach(&x, y, 0.0);
        }
        assert!(cb_mse(&cb0, &mut Rng::new(7)) > 5.0 * train_mse, "the climbing fibre is the teacher");
    }
}
