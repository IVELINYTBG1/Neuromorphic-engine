//! bio_engram (memory #6): an engram = a Hopfield attractor (W = PᵀP/N); recall = pattern completion
//! from a corrupted cue. Ported from bio_engram.py.

use crate::rng::Rng;
use crate::vec;

pub struct Engram {
    n: usize,
    w: Vec<f64>, // n×n
}

impl Engram {
    pub fn new(n: usize) -> Self {
        Engram { n, w: vec![0.0; n * n] }
    }
    pub fn store(&mut self, patterns: &[Vec<f64>]) {
        let n = self.n;
        self.w = vec![0.0; n * n];
        for p in patterns {
            vec::add_outer(&mut self.w, p, p, 1.0 / n as f64);
        }
        for i in 0..n {
            self.w[i * n + i] = 0.0; // no self-connection
        }
    }
    pub fn recall(&self, cue: &[f64]) -> Vec<f64> {
        let mut s = cue.to_vec();
        for _ in 0..20 {
            let a = vec::mv(&self.w, &s, self.n, self.n);
            let nxt: Vec<f64> = a.iter().map(|&x| if x >= 0.0 { 1.0 } else { -1.0 }).collect();
            if nxt == s {
                break;
            }
            s = nxt;
        }
        s
    }
}

pub fn rand_patterns(k: usize, n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    (0..k).map(|_| (0..n).map(|_| (rng.randint(2) as f64) * 2.0 - 1.0).collect()).collect()
}

pub fn corrupt(p: &[f64], frac: f64, rng: &mut Rng) -> Vec<f64> {
    p.iter().map(|&x| if rng.uniform() < frac { -x } else { x }).collect()
}

pub fn overlap(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attractor_completes_a_corrupted_cue() {
        let n = 100;
        let mut g = Rng::new(0);
        let pats = rand_patterns(8, n, &mut g);
        let mut eng = Engram::new(n);
        eng.store(&pats);

        // stored patterns are fixed points
        let min_fixed = (0..8).map(|i| overlap(&eng.recall(&pats[i]), &pats[i])).fold(f64::INFINITY, f64::min);
        assert!(min_fixed > 0.99, "stored patterns are stable fixed points");

        // 20%-corrupted cue pattern-completes to the stored memory
        let cue = corrupt(&pats[0], 0.20, &mut g);
        let rec = eng.recall(&cue);
        assert!(overlap(&rec, &pats[0]) > 0.99 && overlap(&rec, &pats[0]) > overlap(&cue, &pats[0]),
                "pattern completion from a corrupted cue");
    }
}
