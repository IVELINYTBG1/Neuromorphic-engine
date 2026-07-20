//! bio_conj (learning #5, higher-order rung): conjunctive polychronous-group cells — a fixed random
//! expansion + k-WTA makes XOR-of-context linearly separable (100% vs additive 25%). Ported from bio_conj.py.

use crate::rng::Rng;
use crate::vec;

const M: usize = 4; // alphabet: A=0, B=1, S=2, D=3

/// A stream where the next symbol is the EQUALITY (XOR-hard) of two context symbols.
pub fn equality_stream(n_trials: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    let (mut stream, mut r_pos) = (vec![], vec![]);
    for _ in 0..n_trials {
        let (c1, c2) = (rng.randint(2), rng.randint(2));
        let r = if c1 == c2 { 2 } else { 3 };
        r_pos.push(stream.len() + 2);
        stream.extend([c1, c2, r]);
    }
    (stream, r_pos)
}

/// Ring-buffer context: the last k symbols one-hot, most recent first, flattened (k·M).
pub fn build_contexts(stream: &[usize], k: usize) -> Vec<(Vec<f64>, usize)> {
    let mut ring = vec![0.0; k * M];
    let mut pairs = vec![];
    for t in 0..stream.len() - 1 {
        for i in (1..k).rev() {
            for j in 0..M {
                ring[i * M + j] = ring[(i - 1) * M + j];
            }
        }
        for j in 0..M {
            ring[j] = 0.0;
        }
        ring[stream[t]] = 1.0;
        pairs.push((ring.clone(), stream[t + 1]));
    }
    pairs
}

pub fn train_linear(pairs: &[(Vec<f64>, usize)], in_dim: usize, n_out: usize, epochs: usize) -> Vec<f64> {
    let mut w = vec![0.0; n_out * in_dim];
    for _ in 0..epochs {
        for (x, y) in pairs {
            let p = vec::softmax(&vec::mv(&w, x, n_out, in_dim));
            let mut e = p;
            e[*y] -= 1.0;
            vec::add_outer(&mut w, &e, x, -0.2);
        }
    }
    w
}

pub fn linear_predict(w: &[f64], x: &[f64], n_out: usize, in_dim: usize) -> usize {
    vec::argmax(&vec::mv(w, x, n_out, in_dim))
}

pub struct PolyGroupReadout {
    r: Vec<f64>, // n_groups × in_dim (fixed random coincidence weights)
    bias: Vec<f64>,
    in_dim: usize,
    n_groups: usize,
    g_active: usize,
    n_out: usize,
    w: Vec<f64>, // n_out × n_groups (only this learns)
}

impl PolyGroupReadout {
    pub fn new(in_dim: usize, n_out: usize, n_groups: usize, g_active: usize, seed: u64) -> Self {
        let mut g = Rng::new(seed);
        let r = g.randn_vec(n_groups * in_dim, 1.0);
        let bias = g.randn_vec(n_groups, 0.1);
        PolyGroupReadout { r, bias, in_dim, n_groups, g_active, n_out, w: vec![0.0; n_out * n_groups] }
    }
    fn groups(&self, x: &[f64]) -> Vec<f64> {
        let mut proj = vec::mv(&self.r, x, self.n_groups, self.in_dim);
        for i in 0..self.n_groups {
            proj[i] += self.bias[i];
        }
        let mut s = vec![0.0; self.n_groups];
        for &idx in &vec::topk_indices(&proj, self.g_active) {
            s[idx] = 1.0;
        }
        s
    }
    pub fn train(&mut self, pairs: &[(Vec<f64>, usize)], epochs: usize) {
        let feats: Vec<(Vec<f64>, usize)> = pairs.iter().map(|(x, y)| (self.groups(x), *y)).collect();
        for _ in 0..epochs {
            for (f, y) in &feats {
                let p = vec::softmax(&vec::mv(&self.w, f, self.n_out, self.n_groups));
                let mut e = p;
                e[*y] -= 1.0;
                vec::add_outer(&mut self.w, &e, f, -0.1);
            }
        }
    }
    pub fn predict(&self, x: &[f64]) -> usize {
        vec::argmax(&self.logits(x))
    }
    pub fn logits(&self, x: &[f64]) -> Vec<f64> {
        vec::mv(&self.w, &self.groups(x), self.n_out, self.n_groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conjunctive_solves_xor_of_context() {
        let mut g = Rng::new(0);
        let (stream, r_pos) = equality_stream(150, &mut g);
        let pairs = build_contexts(&stream, 2);
        let in_dim = 2 * M;

        let wl = train_linear(&pairs, in_dim, 4, 40);
        let lacc = r_pos.iter().filter(|&&p| {
            let (x, y) = &pairs[p - 1];
            linear_predict(&wl, x, 4, in_dim) == *y
        }).count() as f64 / r_pos.len() as f64;

        let mut cj = PolyGroupReadout::new(in_dim, 4, 200, 12, 1);
        cj.train(&pairs, 40);
        let cacc = r_pos.iter().filter(|&&p| {
            let (x, y) = &pairs[p - 1];
            cj.predict(x) == *y
        }).count() as f64 / r_pos.len() as f64;

        assert!(cacc >= 0.95 && lacc <= 0.65, "conjunctive {} solves XOR-of-context, additive {} can't", cacc, lacc);
    }
}
