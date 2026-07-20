//! bio_camkii (memory #3): CaMKII bistable autophosphorylation switch (Hill n=4) — the 1-bit latch that
//! flips ON with a big Ca²⁺ pulse and PERSISTS after Ca²⁺ is gone; hysteresis; noise-robust. From bio_camkii.py.

use crate::rng::Rng;

pub fn hill(p: f64) -> f64 {
    let pn = p.powi(4);
    pn / (0.5_f64.powi(4) + pn)
}

pub struct CaMKIISwitch {
    k_auto: f64,
    k_deph: f64,
    k_ca: f64,
    dt: f64,
    pub p: f64,
    rng: Rng,
}

impl CaMKIISwitch {
    pub fn new(p: f64) -> Self {
        CaMKIISwitch { k_auto: 1.0, k_deph: 0.25, k_ca: 1.0, dt: 0.2, p, rng: Rng::new(0) }
    }
    pub fn step(&mut self, ca: f64, noise: f64) -> f64 {
        let dp = (self.k_auto * hill(self.p) + self.k_ca * ca) * (1.0 - self.p) - self.k_deph * self.p;
        self.p += self.dt * dp;
        if noise > 0.0 {
            self.p += self.rng.normal() * noise;
        }
        self.p = self.p.clamp(0.0, 1.0);
        self.p
    }
    pub fn relax(&mut self, steps: usize, ca: f64, noise: f64) -> f64 {
        for _ in 0..steps {
            self.step(ca, noise);
        }
        self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bistable_persistent_switch() {
        // 21 initial states settle into two clusters (OFF / ON), nothing stuck between
        let finals: Vec<f64> = (0..21).map(|i| CaMKIISwitch::new(i as f64 / 20.0).relax(400, 0.0, 0.0)).collect();
        let off = finals.iter().filter(|&&f| f < 0.15).count();
        let on = finals.iter().filter(|&&f| f > 0.55).count();
        let middle = finals.iter().filter(|&&f| (0.15..=0.55).contains(&f)).count();
        assert!(off > 0 && on > 0 && middle == 0, "bistable");

        // big Ca²⁺ pulse latches ON and persists; small pulse relaxes OFF
        let mut s = CaMKIISwitch::new(0.0); s.relax(8, 1.0, 0.0);
        assert!(s.relax(400, 0.0, 0.0) > 0.55, "big pulse latches ON & persists");
        let mut s = CaMKIISwitch::new(0.0); s.relax(3, 0.3, 0.0);
        assert!(s.relax(400, 0.0, 0.0) < 0.15, "small pulse relaxes OFF");

        // noise robustness (σ=0.02, 300 steps): ON stays ON, OFF stays OFF
        assert!(CaMKIISwitch::new(0.77).relax(300, 0.0, 0.02) > 0.55);
        assert!(CaMKIISwitch::new(0.0).relax(300, 0.0, 0.02) < 0.15);
    }
}
