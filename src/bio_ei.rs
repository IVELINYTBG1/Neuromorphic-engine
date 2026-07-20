//! bio_ei (thinking #1): excitation/inhibition balance (inhibition-stabilized net). Because the weight
//! matrices are uniform, the network is exactly a 2-variable mean-field (mean E rate, mean I rate).
//! Balanced moderate fixed point; remove inhibition → runaway. From bio_ei.py.

pub struct EINetwork {
    re: f64,
    ri: f64,
    j_ee: f64,
    j_ei: f64,
    j_ie: f64,
    j_ii: f64,
    dt: f64,
    r_max: f64,
    s: f64, // inhibition on (1) / off (0)
}

impl EINetwork {
    pub fn new(inhibition: bool) -> Self {
        EINetwork { re: 0.0, ri: 0.0, j_ee: 3.0, j_ei: 4.0, j_ie: 3.0, j_ii: 2.0,
                    dt: 0.05, r_max: 1.0, s: if inhibition { 1.0 } else { 0.0 } }
    }
    fn phi(&self, x: f64) -> f64 {
        x.clamp(0.0, self.r_max)
    }
    pub fn step(&mut self, i_e: f64) -> f64 {
        let x_e = self.j_ee * self.re - self.s * self.j_ei * self.ri + i_e;
        let x_i = self.j_ie * self.re - self.s * self.j_ii * self.ri;
        self.re += self.dt * (-self.re + self.phi(x_e));
        self.ri += self.dt * (-self.ri + self.phi(x_i));
        self.re
    }
    pub fn settle(&mut self, i_e: f64) -> f64 {
        for _ in 0..500 {
            self.step(i_e);
        }
        self.re
    }
    pub fn drives(&self, i_e: f64) -> (f64, f64) {
        (self.j_ee * self.re + i_e, self.s * self.j_ei * self.ri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_on_a_knifes_edge() {
        let mut net = EINetwork::new(true);
        let re = net.settle(1.0);
        assert!(0.1 < re && re < 0.9, "moderate balanced rate, not silent/saturated");

        net.re += 0.4; // a kick
        let re_kick = net.settle(1.0);
        assert!((re_kick - re).abs() < 1e-3, "returns to the stable attractor");

        let (exc, inh) = net.drives(1.0);
        let net_drive = exc - inh;
        assert!(net_drive.abs() < 0.3 * exc, "net drive is a small residue of the excitation");

        // remove inhibition → runaway saturation
        assert!(EINetwork::new(false).settle(1.0) > 0.95, "no inhibition → runaway");
    }
}
