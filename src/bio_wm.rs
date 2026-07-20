//! bio_wm (memory #5): working memory = persistent recurrent attractor activity, NO weight change.
//! Outlives the cue, rewritable, resettable; no-recurrence control collapses. From bio_wm.py.

use crate::vec;

pub struct WorkingMemory {
    n: usize,
    gain: f64,
    theta: f64,
    w: Vec<f64>, // n×n
    r: Vec<f64>,
}

impl WorkingMemory {
    pub fn new(n: usize, w_self: f64, w_inh: f64) -> Self {
        let mut w = vec![-w_inh; n * n];
        for i in 0..n {
            w[i * n + i] = w_self;
        }
        WorkingMemory { n, gain: 6.0, theta: 0.5, w, r: vec![0.0; n] }
    }
    pub fn reset(&mut self) {
        self.r = vec![0.0; self.n];
    }
    pub fn step(&mut self, inp: &[f64]) {
        let wr = vec::mv(&self.w, &self.r, self.n, self.n);
        for i in 0..self.n {
            self.r[i] = vec::sigmoid(self.gain * (wr[i] + inp[i] - self.theta));
        }
    }
    pub fn run(&mut self, steps: usize, item: Option<usize>, strength: f64) -> Vec<f64> {
        let mut inp = vec![0.0; self.n];
        if let Some(it) = item {
            inp[it] = strength;
        }
        let zero = vec![0.0; self.n];
        for _ in 0..steps {
            self.step(if item.is_some() { &inp } else { &zero });
        }
        self.r.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_activity_no_plasticity() {
        let mut wm = WorkingMemory::new(5, 1.6, 1.2);
        let w_before = wm.w.clone();
        wm.reset();
        wm.run(10, Some(2), 1.0);
        let held = wm.run(40, None, 0.0)[2];
        assert!(held > 0.8, "item held through 40 blank ticks");

        // a stronger new cue overwrites it
        wm.run(15, Some(4), 2.0);
        let after = wm.run(30, None, 0.0);
        assert!(after[4] > 0.8 && after[2] < 0.2, "rewritable");

        // global inhibition clears it
        let inh = vec![-3.0; 5];
        for _ in 0..6 {
            wm.step(&inh);
        }
        let cleared = wm.run(20, None, 0.0);
        assert!(vec::max(&cleared) < 0.2, "reset clears");

        // no-recurrence control collapses
        let mut nr = WorkingMemory::new(5, 0.0, 1.2);
        nr.reset();
        nr.run(10, Some(2), 1.0);
        assert!(nr.run(40, None, 0.0)[2] < 0.2, "no recurrence → collapses");

        // WM is activity, not plasticity — weights never changed
        assert_eq!(w_before, wm.w);
    }
}
