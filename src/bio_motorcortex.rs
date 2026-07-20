//! bio_motorcortex (motor #1): Georgopoulos population vector — a movement direction is a firing-weighted
//! vote of broadly-tuned noisy cells. Precise from a crowd, degrades gracefully under lesion where the
//! single-loudest-neuron readout lurches. Ported from bio_motorcortex.py.

use crate::rng::Rng;
use crate::vec;
use std::f64::consts::PI;

pub struct MotorCortex {
    pub n: usize,
    pd: Vec<f64>,
    u: Vec<(f64, f64)>,
    baseline: f64,
    modulation: f64,
    noise: f64,
    rng: Rng,
}

impl MotorCortex {
    pub fn new(seed: u64) -> Self {
        let n = 120;
        let pd: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
        let u: Vec<(f64, f64)> = pd.iter().map(|&p| (p.cos(), p.sin())).collect();
        MotorCortex { n, pd, u, baseline: 1.0, modulation: 1.0, noise: 0.4, rng: Rng::new(seed) }
    }
    fn rates(&mut self, theta: f64, mask: Option<&[f64]>) -> Vec<f64> {
        let mut r = vec![0.0; self.n];
        for i in 0..self.n {
            let base = (self.baseline + self.modulation * (theta - self.pd[i]).cos()).max(0.0);
            r[i] = (base + self.noise * self.rng.normal()).max(0.0);
            if let Some(m) = mask {
                r[i] *= m[i];
            }
        }
        r
    }
    /// (population-vector estimate, labeled-line estimate)
    pub fn decode(&mut self, theta: f64, mask: Option<&[f64]>) -> (f64, f64) {
        let r = self.rates(theta, mask);
        let (mut vx, mut vy) = (0.0, 0.0);
        for i in 0..self.n {
            vx += r[i] * self.u[i].0;
            vy += r[i] * self.u[i].1;
        }
        let pv = vy.atan2(vx);
        let ll = self.pd[vec::argmax(&r)];
        (pv, ll)
    }
    pub fn tuning_rate(&self, theta: f64, neuron: usize) -> f64 {
        self.baseline + self.modulation * (theta - self.pd[neuron]).cos()
    }
}

pub fn ang_err(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % (2.0 * PI);
    d.min(2.0 * PI - d)
}

/// mean population-vector and labeled-line error in DEGREES over target directions
pub fn mean_errors(mc: &mut MotorCortex, targets: &[f64], mask: Option<&[f64]>) -> (f64, f64) {
    let (mut pv_e, mut ll_e) = (0.0, 0.0);
    for &t in targets {
        let (pv, ll) = mc.decode(t, mask);
        pv_e += ang_err(pv, t);
        ll_e += ang_err(ll, t);
    }
    let n = targets.len() as f64;
    ((pv_e / n).to_degrees(), (ll_e / n).to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn population_vector_decodes() {
        let mut mc = MotorCortex::new(0);
        let targets: Vec<f64> = (0..36).map(|i| 2.0 * PI * i as f64 / 36.0).collect();
        let (pv_deg, ll_deg) = mean_errors(&mut mc, &targets, None);
        assert!(pv_deg < 6.0, "population vector decodes precisely ({}°)", pv_deg);
        assert!(pv_deg < 0.5 * ll_deg, "the crowd beats the single loudest neuron");

        // graceful degradation: beats labeled-line through 75% lesion, stays accurate
        let mut worst_pv = 0.0f64;
        let mut beats = true;
        for &frac in &[0.0, 0.25, 0.5, 0.75] {
            let keep: Vec<f64> = (0..mc.n).map(|_| if mc.rng.uniform() >= frac { 1.0 } else { 0.0 }).collect();
            let (p, l) = mean_errors(&mut mc, &targets, Some(&keep));
            worst_pv = worst_pv.max(p);
            if p >= l {
                beats = false;
            }
        }
        assert!(beats && worst_pv < 15.0, "graceful through 75% lesion");

        // one neuron is ambiguous (cosine many-to-one), the population distinguishes
        let th_a = mc.tuning_rate_pd(0) + 1.0;
        let th_b = mc.tuning_rate_pd(0) - 1.0;
        assert!((mc.tuning_rate(th_a, 0) - mc.tuning_rate(th_b, 0)).abs() < 1e-6, "single neuron ambiguous");
        let (pop_a, _) = mc.decode(th_a, None);
        let (pop_b, _) = mc.decode(th_b, None);
        assert!(ang_err(pop_a, pop_b) > 1.0, "population distinguishes what one neuron can't");
    }
}

impl MotorCortex {
    fn tuning_rate_pd(&self, neuron: usize) -> f64 {
        self.pd[neuron]
    }
}
