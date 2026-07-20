//! bio_branch (learning #5, branching/Markov): the softmax readout IS P(next|context); a local delta rule
//! + Polyak–Ruppert averaging settles it on the empirical conditional, and generation SAMPLES (not argmax)
//! so it reproduces a branching source. Ported from bio_branch.py.

use crate::bio_poly::BioPolyChron;
use crate::rng::Rng;
use crate::vec;

pub struct MarkovSource {
    p: Vec<Vec<f64>>, // m × m, rows = P(next | cur)
    m: usize,
}
impl MarkovSource {
    pub fn new(p: Vec<Vec<f64>>) -> Self {
        let m = p.len();
        MarkovSource { p, m }
    }
    pub fn sample_stream(&self, n: usize, start: usize, rng: &mut Rng) -> Vec<usize> {
        let mut out = vec![start];
        let mut cur = start;
        for _ in 0..n - 1 {
            cur = rng.multinomial(&self.p[cur]);
            out.push(cur);
        }
        out
    }
}

/// empirical transition matrix estimated from a stream
pub fn estimate_matrix(stream: &[usize], m: usize) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0.0; m]; m];
    for t in 0..stream.len() - 1 {
        counts[stream[t]][stream[t + 1]] += 1.0;
    }
    for row in counts.iter_mut() {
        let s: f64 = row.iter().sum::<f64>() + 1e-9;
        row.iter_mut().for_each(|c| *c /= s);
    }
    counts
}

pub struct MarkovPolyChron {
    net: BioPolyChron,
}

impl MarkovPolyChron {
    pub fn new(n_symbols: usize, k: usize) -> Self {
        MarkovPolyChron { net: BioPolyChron::new(n_symbols, k) }
    }
    /// local delta rule + Polyak–Ruppert tail-averaging → the empirical conditional (low-variance MLE)
    pub fn teach_stream(&mut self, seq: &[usize], epochs: usize, lr: f64, avg_tail: usize, rng: &mut Rng) {
        let mut pairs: Vec<(Vec<f64>, usize)> = vec![];
        self.net.reset_state();
        for t in 0..seq.len() - 1 {
            self.net.observe(seq[t]);
            pairs.push((self.net.ring().to_vec(), seq[t + 1]));
        }
        self.net.lr = lr;
        let dim = self.net.ctx_dim();
        let m = self.net.n_symbols();
        let mut w_sum = vec![0.0; m * dim];
        let mut n = 0;
        for e in 0..epochs {
            for j in rng.randperm(pairs.len()) {
                let (ctx, y) = &pairs[j];
                let p = vec::softmax(&vec::mv(&self.net.w, ctx, m, dim));
                let mut err = p;
                err[*y] -= 1.0;
                vec::add_outer(&mut self.net.w, &err, ctx, -self.net.lr);
            }
            if e >= epochs - avg_tail {
                for i in 0..w_sum.len() {
                    w_sum[i] += self.net.w[i];
                }
                n += 1;
            }
        }
        for i in 0..w_sum.len() {
            self.net.w[i] = w_sum[i] / n as f64;
        }
    }
    pub fn dist(&mut self, history: &[usize]) -> Vec<f64> {
        self.net.reset_state();
        for &s in history {
            self.net.observe(s);
        }
        vec::softmax(&self.net.logits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_and_samples_a_branching_source() {
        let src = MarkovSource::new(vec![
            vec![0.0, 0.7, 0.3],
            vec![0.4, 0.0, 0.6],
            vec![1.0, 0.0, 0.0],
        ]);
        let mut g = Rng::new(0);
        let stream = src.sample_stream(1200, 0, &mut g);
        let mut net = MarkovPolyChron::new(3, 1);
        let mut gt = Rng::new(0);
        net.teach_stream(&stream, 20, 0.05, 8, &mut gt);
        let emp = estimate_matrix(&stream, 3);

        // learns the conditional distribution to match the empirical MLE
        let learn_err = (0..3).map(|i| {
            let d = net.dist(&[i]);
            (0..3).map(|j| (d[j] - emp[i][j]).abs()).fold(0.0, f64::max)
        }).fold(0.0, f64::max);
        assert!(learn_err < 0.05, "learns the branching distribution (err {})", learn_err);

        // P(next | 'a') genuinely branches over b and c (not collapsed to the mode)
        let pa = net.dist(&[0]);
        assert!(pa[1] > 0.1 && pa[2] > 0.1, "samples a branch, no collapse");
    }
}
