//! bio_drive (limbic #4): homeostatic drive — a need is a set-point deviation, reward is the drive it
//! reduces, and the same resource is wanted only when lacked (alliesthesia). Ported from bio_drive.py.

pub struct Homeostat {
    pub setpoint: f64,
    pub state: f64,
    pub leak: f64,
}

impl Default for Homeostat {
    fn default() -> Self {
        Homeostat { setpoint: 1.0, state: 1.0, leak: 0.1 }
    }
}

impl Homeostat {
    pub fn drive(&self) -> f64 {
        (self.setpoint - self.state).abs()
    }
    pub fn metabolize(&mut self) {
        self.state -= self.leak;
    }
    /// reward = the reduction in drive (not the substance)
    pub fn consume(&mut self, amount: f64) -> f64 {
        let before = self.drive();
        self.state += amount;
        before - self.drive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motivation_from_homeostasis() {
        // deviation from set-point creates drive
        let mut h = Homeostat::default();
        let d0 = h.drive();
        for _ in 0..5 {
            h.metabolize();
        }
        assert!(d0 < 0.05 && h.drive() > 0.3, "a need should grow from metabolism");

        // reward = drive reduction
        assert!(h.consume(0.4) > 0.1, "consuming while hungry should reward");

        // the drive regulates the state (stays alive)
        let mut h = Homeostat::default();
        let mut trace = vec![];
        for _ in 0..80 {
            h.metabolize();
            if h.drive() > 0.2 && h.state < h.setpoint {
                h.consume(0.4);
            }
            trace.push(h.state);
        }
        let steady = &trace[20..];
        let (lo, hi) = (
            steady.iter().cloned().fold(f64::INFINITY, f64::min),
            steady.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        assert!(lo > 0.4 && hi < 1.3, "should stay bounded near the set-point");

        // alliesthesia: same bite → rewarding when starving, aversive when full
        let mut starving = Homeostat { state: 0.2, ..Default::default() };
        let mut full = Homeostat { state: 1.0, ..Default::default() };
        assert!(starving.consume(0.4) > 0.1 && full.consume(0.4) < -0.1,
                "incentive depends on inner need (alliesthesia)");
    }
}
