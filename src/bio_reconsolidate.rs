//! bio_reconsolidate (memory #8): retrieval reopens lability — recall→update rewrites; the same update
//! without recall doesn't take; an amnestic hit erases a REACTIVATED memory but spares a stable one.
//! Ported from bio_reconsolidate.py.

use crate::rng::Rng;
use crate::vec;

pub struct ReconsolidatingMemory {
    d: usize,
    w: Vec<f64>, // d×d
    base_lr: f64,
    labile_gain: f64,
    window_decay: f64,
    labile: f64,
}

impl ReconsolidatingMemory {
    pub fn new(d: usize) -> Self {
        ReconsolidatingMemory { d, w: vec![0.0; d * d], base_lr: 0.30, labile_gain: 8.0,
                                window_decay: 0.55, labile: 0.0 }
    }
    pub fn store(&mut self, cue: &[f64], target: &[f64], lr: f64) {
        vec::add_outer(&mut self.w, target, cue, lr);
    }
    pub fn peek(&self, cue: &[f64]) -> Vec<f64> {
        vec::mv(&self.w, cue, self.d, self.d)
            .iter()
            .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
            .collect()
    }
    pub fn recall(&mut self, cue: &[f64]) -> Vec<f64> {
        self.labile = 1.0;
        self.peek(cue)
    }
    pub fn tick(&mut self, n: usize) {
        for _ in 0..n {
            self.labile *= self.window_decay;
        }
    }
    pub fn update(&mut self, cue: &[f64], new_target: &[f64]) {
        let lr = self.base_lr * (1.0 + self.labile_gain * self.labile);
        vec::add_outer(&mut self.w, new_target, cue, lr);
    }
    pub fn disrupt(&mut self, cue: &[f64]) {
        let target_now = self.peek(cue);
        vec::add_outer(&mut self.w, &target_now, cue, -self.labile * 2.0);
    }
}

pub fn overlap(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_reopens_lability() {
        let d = 120;
        let mut g = Rng::new(0);
        let pm1: Vec<f64> = (0..d).map(|_| (g.randint(2) as f64) * 2.0 - 1.0).collect();
        let t_old: Vec<f64> = (0..d).map(|_| (g.randint(2) as f64) * 2.0 - 1.0).collect();
        let t_new: Vec<f64> = (0..d).map(|_| (g.randint(2) as f64) * 2.0 - 1.0).collect();
        let cue = pm1;
        let fresh = |cue: &[f64], t_old: &[f64]| {
            let mut m = ReconsolidatingMemory::new(d);
            m.store(cue, t_old, 2.0);
            m
        };

        // recall → update rewrites the memory
        let mut a = fresh(&cue, &t_old);
        a.recall(&cue);
        a.update(&cue, &t_new);
        assert!(overlap(&a.peek(&cue), &t_new) > 0.9 && overlap(&a.peek(&cue), &t_old) < 0.7,
                "recall→update rewrites");

        // the same update WITHOUT recall doesn't take
        let mut b = fresh(&cue, &t_old);
        b.update(&cue, &t_new);
        assert!(overlap(&b.peek(&cue), &t_old) > 0.9, "no recall → memory preserved");

        // update after the labile window closes fails
        let mut c = fresh(&cue, &t_old);
        c.recall(&cue);
        c.tick(6);
        c.update(&cue, &t_new);
        assert!(overlap(&c.peek(&cue), &t_old) > 0.9, "window closed → update fails");

        // amnestic hit erases a REACTIVATED memory but spares a stable one
        let mut d1 = fresh(&cue, &t_old);
        d1.recall(&cue);
        d1.disrupt(&cue);
        let mut d2 = fresh(&cue, &t_old);
        d2.disrupt(&cue);
        assert!(overlap(&d1.peek(&cue), &t_old) < 0.7 && overlap(&d2.peek(&cue), &t_old) > 0.9,
                "reactivated erased, stable spared");
    }
}
