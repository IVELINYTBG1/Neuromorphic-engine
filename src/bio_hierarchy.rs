//! bio_hierarchy (sensing #3): Hubel–Wiesel simple→complex cells. Gabor simple cells are orientation-tuned
//! but phase-sensitive; complex cells (quadrature energy) are phase/position-INVARIANT while keeping
//! orientation tuning — invariance from one pooling step. Ported from bio_hierarchy.py.

use crate::vec;
use std::f64::consts::PI;

const SIZE: usize = 25;

fn field<F: Fn(f64, f64) -> f64>(f: F) -> Vec<f64> {
    let c = (SIZE as f64 - 1.0) / 2.0;
    let mut v = vec![0.0; SIZE * SIZE];
    for i in 0..SIZE {
        for j in 0..SIZE {
            v[i * SIZE + j] = f(j as f64 - c, i as f64 - c); // (x, y)
        }
    }
    v
}

pub fn gabor(theta: f64, phase: f64) -> Vec<f64> {
    let (lam, sigma, gamma) = (8.0, 8.0, 0.5);
    let mut g = field(|x, y| {
        let xr = x * theta.cos() + y * theta.sin();
        let yr = -x * theta.sin() + y * theta.cos();
        let env = (-(xr * xr + (gamma * yr).powi(2)) / (2.0 * sigma * sigma)).exp();
        env * (2.0 * PI * xr / lam + phase).cos()
    });
    let m = vec::mean(&g);
    g.iter_mut().for_each(|v| *v -= m);
    let n = vec::norm(&g);
    g.iter_mut().for_each(|v| *v /= n);
    g
}

pub fn grating(theta: f64, phase: f64) -> Vec<f64> {
    let lam = 8.0;
    let mut s = field(|x, y| {
        let xr = x * theta.cos() + y * theta.sin();
        (2.0 * PI * xr / lam + phase).cos()
    });
    let m = vec::mean(&s);
    s.iter_mut().for_each(|v| *v -= m);
    s
}

pub fn simple(stim: &[f64], theta: f64, phase: f64) -> f64 {
    vec::relu(vec::dot(&gabor(theta, phase), stim))
}

pub fn complex_cell(stim: &[f64], theta: f64) -> f64 {
    let e0 = vec::dot(&gabor(theta, 0.0), stim);
    let e90 = vec::dot(&gabor(theta, PI / 2.0), stim);
    (e0 * e0 + e90 * e90).sqrt()
}

fn cv(values: &[f64]) -> f64 {
    let m = vec::mean(values);
    let sd = (values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    sd / (m + 1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_to_complex_invariance() {
        let (pref, orth) = (0.0, PI / 2.0);
        let phases: Vec<f64> = (0..8).map(|i| i as f64 * PI / 4.0).collect();

        // simple cell: orientation-tuned but phase-sensitive
        assert!(simple(&grating(pref, 0.0), pref, 0.0) > 3.0 * (simple(&grating(orth, 0.0), pref, 0.0) + 1e-6),
                "simple cell orientation-selective");
        let s_by_phase: Vec<f64> = phases.iter().map(|&ph| simple(&grating(pref, ph), pref, 0.0)).collect();
        assert!(cv(&s_by_phase) > 0.4, "simple cell phase-sensitive");

        // complex cell: phase-INVARIANT, still orientation-tuned
        let c_by_phase: Vec<f64> = phases.iter().map(|&ph| complex_cell(&grating(pref, ph), pref)).collect();
        assert!(cv(&c_by_phase) < 0.1, "complex cell phase-invariant");
        let c_pref: f64 = phases.iter().map(|&ph| complex_cell(&grating(pref, ph), pref)).sum::<f64>() / 8.0;
        let c_orth: f64 = phases.iter().map(|&ph| complex_cell(&grating(orth, ph), pref)).sum::<f64>() / 8.0;
        assert!(c_pref > 3.0 * (c_orth + 1e-6), "complex cell keeps orientation tuning");
    }
}
