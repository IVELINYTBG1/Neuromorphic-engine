//! bio_workspace (thinking #6): global-workspace ignition (Dehaene) — all-or-none threshold, broadcast
//! only above it, one-at-a-time bottleneck, self-sustaining then resettable. From bio_workspace.py.

use crate::vec;

pub struct GlobalWorkspace {
    n: usize,
    gain: f64,
    w_self: f64,
    w_inh: f64,
    theta: f64,
    dt: f64,
    steps: usize,
}

impl Default for GlobalWorkspace {
    fn default() -> Self {
        GlobalWorkspace { n: 4, gain: 12.0, w_self: 1.2, w_inh: 2.0, theta: 0.6, dt: 0.2, steps: 200 }
    }
}

impl GlobalWorkspace {
    /// A workspace sized for `n` competing thoughts (all other dynamics as Default).
    pub fn with_n(n: usize) -> Self {
        GlobalWorkspace { n, ..GlobalWorkspace::default() }
    }

    pub fn run(&self, stim: &[f64], a0: Option<&[f64]>, reset: bool) -> Vec<f64> {
        let mut a = match a0 {
            Some(x) => x.to_vec(),
            None => vec![0.0; self.n],
        };
        for _ in 0..self.steps {
            let total: f64 = a.iter().sum();
            let mut na = a.clone();
            for i in 0..self.n {
                let mut drive = stim[i] + self.w_self * a[i] - self.w_inh * (total - a[i]) - self.theta;
                if reset {
                    drive -= 10.0;
                }
                na[i] = a[i] + self.dt * (-a[i] + vec::sigmoid(self.gain * drive));
            }
            a = na;
        }
        a
    }
    pub fn broadcast(a: &[f64]) -> f64 {
        vec::max(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_or_none_ignition() {
        let ws = GlobalWorkspace::default();

        // all-or-none: sweep the stimulus; activity is off or on, ~no forbidden middle
        let sweep: Vec<f64> = (0..21).map(|i| ws.run(&[i as f64 / 20.0, 0.0, 0.0, 0.0], None, false)[0]).collect();
        let off = sweep.iter().filter(|&&a| a < 0.2).count();
        let on = sweep.iter().filter(|&&a| a > 0.8).count();
        let middle = sweep.iter().filter(|&&a| (0.2..=0.8).contains(&a)).count();
        assert!(off > 0 && on > 0 && middle <= 2, "all-or-none step");

        // broadcast only above threshold
        let g_sub = GlobalWorkspace::broadcast(&ws.run(&[0.3, 0.0, 0.0, 0.0], None, false));
        let g_sup = GlobalWorkspace::broadcast(&ws.run(&[0.8, 0.0, 0.0, 0.0], None, false));
        assert!(g_sub < 0.1 && g_sup > 0.8, "subliminal vs conscious broadcast");

        // one-at-a-time bottleneck
        let a_two = ws.run(&[1.0, 0.95, 0.0, 0.0], None, false);
        assert_eq!(a_two.iter().filter(|&&x| x > 0.5).count(), 1, "bottleneck: one ignites");

        // self-sustains then resets
        let ignited = ws.run(&[1.0, 0.0, 0.0, 0.0], None, false);
        let held = ws.run(&[0.0; 4], Some(&ignited), false);
        let cleared = ws.run(&[0.0; 4], Some(&held), true);
        assert!(held[0] > 0.8 && cleared[0] < 0.2, "sustains then resettable");
    }
}
