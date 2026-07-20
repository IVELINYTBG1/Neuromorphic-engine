//! bio_nmda (memory #1): NMDA Ca²⁺ as a coincidence AND-gate (glutamate AND depolarization, Mg²⁺ block)
//! + Ca²⁺-driven AMPAR trafficking (weight = receptor count). Ported from bio_nmda.py.

/// Mg²⁺ plug expelled by postsynaptic depolarization (Jahr–Stevens-style sigmoid). Single source of truth.
pub fn mg_unblock(v_post: f64) -> f64 {
    1.0 / (1.0 + (-(v_post - 0.5) / 0.12).exp())
}

pub struct NMDASynapse {
    pub w: f64,
    theta_d: f64,
    theta_p: f64,
    ltp: f64,
    ltd: f64,
    wmax: f64,
}

impl Default for NMDASynapse {
    fn default() -> Self {
        NMDASynapse { w: 0.5, theta_d: 0.35, theta_p: 0.60, ltp: 0.05, ltd: 0.03, wmax: 2.0 }
    }
}

impl NMDASynapse {
    pub fn with_w(w: f64) -> Self {
        NMDASynapse { w, ..Default::default() }
    }
    pub fn calcium(&self, glutamate: f64, v_post: f64) -> f64 {
        glutamate * mg_unblock(v_post)
    }
    pub fn transmit(&mut self, glutamate: f64, v_post: f64) -> f64 {
        let ampa = glutamate * self.w;
        let ca = self.calcium(glutamate, v_post);
        if ca >= self.theta_p {
            self.w = (self.w + self.ltp * ca).min(self.wmax);
        } else if ca >= self.theta_d {
            self.w = (self.w - self.ltd).max(0.0);
        }
        ampa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nmda_and_gate_and_trafficking() {
        let s = NMDASynapse::default();
        // Ca²⁺ AND-gate: only glutamate AND depolarization
        assert!(s.calcium(1.0, 1.0) > 0.8);
        assert!(s.calcium(1.0, 0.0) < 0.2);
        assert!(s.calcium(0.0, 1.0) < 0.2);
        assert!(s.calcium(0.0, 0.0) < 0.2);

        // Mg²⁺ unblock monotone in depolarization
        let vs = [0.0, 0.25, 0.5, 0.75, 1.0];
        let ub: Vec<f64> = vs.iter().map(|&v| mg_unblock(v)).collect();
        assert!((0..ub.len() - 1).all(|i| ub[i] < ub[i + 1]));

        // high Ca²⁺ → LTP, mid → LTD, low → no change
        let mut s = NMDASynapse::with_w(0.5);
        let w0 = s.w;
        for _ in 0..20 { s.transmit(1.0, 1.0); }
        assert!(s.w > w0 + 0.1, "LTP raises the weight");
        let mut s = NMDASynapse::with_w(1.5);
        for _ in 0..20 { s.transmit(1.0, 0.50); }
        assert!(s.w < 1.5 - 0.05, "LTD lowers the weight");
        let mut s = NMDASynapse::with_w(1.0);
        for _ in 0..20 { s.transmit(1.0, 0.15); }
        assert!((s.w - 1.0).abs() < 0.05, "low Ca²⁺ → no change");
    }
}
