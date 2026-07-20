//! bio_neuromod (limbic #6): one neuromodulator M (the neuromodulator) sets plasticity + gain + exploration at once
//! → flips exploit↔explore. Ported from bio_neuromod.py.

use crate::vec;

pub struct NeuroModulator {
    pub base_lr: f64,
    pub base_gain: f64,
    pub base_temp: f64,
    pub k: f64,
}

impl Default for NeuroModulator {
    fn default() -> Self {
        NeuroModulator { base_lr: 0.05, base_gain: 2.0, base_temp: 0.25, k: 4.0 }
    }
}

impl NeuroModulator {
    pub fn lr(&self, m: f64) -> f64 {
        self.base_lr * (1.0 + self.k * m)
    }
    pub fn gain(&self, m: f64) -> f64 {
        self.base_gain * (1.0 + self.k * m)
    }
    pub fn temperature(&self, m: f64) -> f64 {
        self.base_temp * (1.0 + self.k * m)
    }
}

fn entropy(p: &[f64]) -> f64 {
    -p.iter().map(|&x| x * (x + 1e-12).ln()).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_scalar_flips_regime() {
        let nm = NeuroModulator::default();
        let (lo, hi) = (0.1, 0.9);

        // plasticity: learn a target with the M-scaled rate; high M converges faster
        let learn_err = |m: f64| {
            let mut w = 0.0;
            for _ in 0..8 {
                w += nm.lr(m) * (1.0 - w);
            }
            (1.0 - w).abs()
        };
        assert!(learn_err(hi) < 0.5 * learn_err(lo), "plasticity should rise with M");

        // gain: a fixed small input difference → sharper output difference at high M
        let contrast = |m: f64| {
            let g = nm.gain(m);
            vec::sigmoid(g * 0.1) - vec::sigmoid(g * -0.1)
        };
        assert!(contrast(hi) > 2.0 * contrast(lo), "gain should sharpen with M");

        // exploration + regime switch
        let values = [1.0, 0.5, 0.3, 0.1];
        let scaled = |t: f64| values.iter().map(|v| v / t).collect::<Vec<f64>>();
        let p_lo = vec::softmax(&scaled(nm.temperature(lo)));
        let p_hi = vec::softmax(&scaled(nm.temperature(hi)));
        assert!(entropy(&p_hi) > entropy(&p_lo) + 0.3, "policy should gamble more at high M");
        assert!(vec::max(&p_lo) > 0.6 && vec::max(&p_hi) < 0.45, "one scalar flips exploit → explore");
    }
}
