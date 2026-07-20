//! bio_select (thinking #4): basal-ganglia action selection by focused disinhibition; dopamine is the
//! gain on decisiveness (akinesia ↔ clean pick-one ↔ impulsivity). Ported from bio_select.py.

use crate::vec;

pub struct BasalGanglia {
    nogo: f64,
    w_lat: f64,
    dt: f64,
    steps: usize,
    thresh: f64,
}

impl Default for BasalGanglia {
    fn default() -> Self {
        BasalGanglia { nogo: 0.45, w_lat: 1.2, dt: 0.2, steps: 200, thresh: 0.05 }
    }
}

impl BasalGanglia {
    pub fn new(steps: usize) -> Self {
        BasalGanglia { steps, ..Default::default() }
    }
    /// Returns (released activity per channel, selected mask).
    pub fn select(&self, salience: &[f64], da: f64) -> (Vec<f64>, Vec<bool>) {
        let n = salience.len();
        let go: Vec<f64> = salience.iter().map(|s| da * s).collect();
        let brake = self.nogo / da;
        let mut r = vec![0.0; n];
        for _ in 0..self.steps {
            let total: f64 = r.iter().sum();
            for i in 0..n {
                let lateral = self.w_lat * (total - r[i]);
                let target = vec::relu(go[i] - brake - lateral).min(1.0);
                r[i] += self.dt * (-r[i] + target);
            }
        }
        let sel: Vec<bool> = r.iter().map(|&x| x > self.thresh).collect();
        (r, sel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n_sel(sel: &[bool]) -> usize {
        sel.iter().filter(|&&s| s).count()
    }

    #[test]
    fn dopamine_gates_selection() {
        let bg = BasalGanglia::default();
        let salience = [0.9, 0.6, 0.4, 0.2];

        // normal DA → exactly one winner, the most salient
        let (r1, sel1) = bg.select(&salience, 1.0);
        assert_eq!(n_sel(&sel1), 1, "one winner at normal DA");
        assert!(vec::argmax(&r1) == 0 && sel1[0], "the winner is the most salient");

        // low DA → akinesia (nothing selected)
        assert_eq!(n_sel(&bg.select(&salience, 0.3).1), 0, "low DA → akinesia");

        // high DA → impulsivity (more than one selected)
        assert!(n_sel(&bg.select(&salience, 3.0).1) > 1, "high DA → impulsive");

        // near-tie still resolves to one
        assert_eq!(n_sel(&bg.select(&[0.80, 0.78, 0.3], 1.0).1), 1, "decisive near a tie");
    }
}
