//! bio_poly (learning #5, repeated symbols): polychronization — axonal delay taps turn a temporal context
//! into a spatial coincidence, so repeated symbols ("hello") are disambiguated. Delay depth = context
//! order. Local single-layer delta rule, no backprop. Ported from bio_poly.py.

use crate::vec;

pub struct BioPolyChron {
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub lr: f64,
    pub w: Vec<f64>,        // m × (m·k)  — pub so bio_branch can Polyak-average it
    pub(crate) ring: Vec<f64>, // k × m  (delay line, most recent at row 0)
}

impl BioPolyChron {
    pub fn new(n_symbols: usize, k: usize) -> Self {
        BioPolyChron { m: n_symbols, k, lr: 0.5, w: vec![0.0; n_symbols * n_symbols * k], ring: vec![0.0; k * n_symbols] }
    }
    pub fn reset_state(&mut self) {
        self.ring = vec![0.0; self.k * self.m];
    }
    pub fn observe(&mut self, sym: usize) {
        for i in (1..self.k).rev() {
            for j in 0..self.m {
                self.ring[i * self.m + j] = self.ring[(i - 1) * self.m + j];
            }
        }
        for j in 0..self.m {
            self.ring[j] = 0.0;
        }
        self.ring[sym] = 1.0;
    }
    pub fn logits(&self) -> Vec<f64> {
        vec::mv(&self.w, &self.ring, self.m, self.m * self.k)
    }
    pub fn ring(&self) -> &[f64] {
        &self.ring
    }
    pub fn n_symbols(&self) -> usize {
        self.m
    }
    pub fn ctx_dim(&self) -> usize {
        self.m * self.k
    }
    pub fn teach(&mut self, seq: &[usize], epochs: usize) {
        let mut pairs = vec![];
        self.reset_state();
        for t in 0..seq.len() - 1 {
            self.observe(seq[t]);
            pairs.push((self.ring.clone(), seq[t + 1]));
        }
        for _ in 0..epochs {
            for (ctx, y) in &pairs {
                let p = vec::softmax(&vec::mv(&self.w, ctx, self.m, self.m * self.k));
                let mut err = p;
                err[*y] -= 1.0;
                vec::add_outer(&mut self.w, &err, ctx, -self.lr);
            }
        }
    }
    pub fn recall(&mut self, cue: usize, steps: usize) -> Vec<usize> {
        self.reset_state();
        self.observe(cue);
        let mut out = vec![cue];
        for _ in 0..steps {
            let nxt = vec::argmax(&self.logits());
            out.push(nxt);
            self.observe(nxt);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(word: &str) -> (Vec<usize>, usize) {
        let mut alpha: Vec<char> = word.chars().collect();
        alpha.sort();
        alpha.dedup();
        let seq: Vec<usize> = word.chars().map(|c| alpha.iter().position(|&a| a == c).unwrap()).collect();
        (seq, alpha.len())
    }

    #[test]
    fn axonal_delays_disambiguate_repeats() {
        let (seq, m) = encode("hello"); // repeated 'l' needs delay depth ≥ 2
        let mut flat = BioPolyChron::new(m, 1); // k=1: no delays → collapses
        flat.teach(&seq, 400);
        let mut deep = BioPolyChron::new(m, 2); // k=2: delay line disambiguates
        deep.teach(&seq, 400);
        assert!(flat.recall(seq[0], seq.len() - 1) != seq, "k=1 collapses on repeats");
        assert!(deep.recall(seq[0], seq.len() - 1) == seq, "delays recall 'hello'");
    }
}
