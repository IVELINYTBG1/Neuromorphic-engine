//! bio_ternary (learning #8): ternary/1-bit weights (BitNet-style) — weights ∈ {-1,0,+1}, multiply-free,
//! ~half-sparse, still generalizes (with an honest quantization tax). DFA + straight-through. From bio_ternary.py.

use crate::bio_scale::make_checkerboard;
use crate::rng::Rng;
use crate::vec;

/// per-row absmean scale; returns (ternary weights ∈ {-1,0,1}, scale per row)
fn ternarize(w: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut scale = vec![0.0; rows];
    for i in 0..rows {
        let s: f64 = (0..cols).map(|j| w[i * cols + j].abs()).sum::<f64>() / cols as f64 + 1e-8;
        scale[i] = s;
    }
    let wq: Vec<f64> = (0..rows * cols)
        .map(|idx| {
            let i = idx / cols;
            (w[idx] / scale[i]).round().clamp(-1.0, 1.0)
        })
        .collect();
    (wq, scale)
}

pub struct TernaryBioNet {
    n_in: usize,
    n_hid: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    bfb: Vec<f64>,
    lr: f64,
    x: Vec<f64>,
    h: Vec<f64>,
    y: Vec<f64>,
}

impl TernaryBioNet {
    pub fn new(n_in: usize, n_hid: usize, n_out: usize, rng: &mut Rng) -> Self {
        TernaryBioNet {
            n_in, n_hid,
            w1: rng.randn_vec(n_hid * n_in, 0.5),
            b1: vec![0.0; n_hid],
            w2: rng.randn_vec(n_out * n_hid, 0.5),
            b2: vec![0.0; n_out],
            bfb: rng.randn_vec(n_hid * n_out, 1.0),
            lr: 0.15,
            x: vec![], h: vec![], y: vec![],
        }
    }
    pub fn forward(&mut self, x: &[f64]) -> Vec<f64> {
        self.x = x.to_vec();
        let n_out = self.b2.len();
        let (w1q, s1) = ternarize(&self.w1, self.n_hid, self.n_in);
        let (w2q, s2) = ternarize(&self.w2, n_out, self.n_hid);
        let a1 = vec::mv(&w1q, x, self.n_hid, self.n_in);
        self.h = (0..self.n_hid).map(|i| (s1[i] * a1[i] + self.b1[i]).tanh()).collect();
        let a2 = vec::mv(&w2q, &self.h, n_out, self.n_hid);
        self.y = (0..n_out).map(|i| vec::sigmoid(s2[i] * a2[i] + self.b2[i])).collect();
        self.y.clone()
    }
    pub fn learn(&mut self, target: &[f64]) {
        let n_out = self.b2.len();
        let e: Vec<f64> = (0..n_out).map(|i| self.y[i] - target[i]).collect();
        vec::add_outer(&mut self.w2, &e, &self.h, -self.lr); // straight-through: update LATENT weights
        for i in 0..n_out {
            self.b2[i] -= self.lr * e[i];
        }
        let be = vec::mv(&self.bfb, &e, self.n_hid, n_out);
        let dh: Vec<f64> = (0..self.n_hid).map(|i| be[i] * (1.0 - self.h[i] * self.h[i])).collect();
        vec::add_outer(&mut self.w1, &dh, &self.x, -self.lr);
        for i in 0..self.n_hid {
            self.b1[i] -= self.lr * dh[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_multiply_free_still_generalizes() {
        let mut g = Rng::new(0);
        let (xtr, ytr) = make_checkerboard(1000, &mut g);
        let (xte, yte) = make_checkerboard(500, &mut g);
        let mut net = TernaryBioNet::new(2, 256, 1, &mut g);
        for _ in 0..600 {
            for &i in &g.randperm(xtr.len()) {
                net.forward(&xtr[i]);
                net.learn(&[ytr[i]]);
            }
        }
        let te = (0..xte.len()).filter(|&i| (net.forward(&xte[i])[0] > 0.5) == (yte[i] > 0.5)).count() as f64 / xte.len() as f64;

        // weights are ternary and sparse
        let (w1q, _) = ternarize(&net.w1, 256, 2);
        let (w2q, _) = ternarize(&net.w2, 1, 256);
        assert!(w1q.iter().chain(&w2q).all(|&v| v == -1.0 || v == 0.0 || v == 1.0), "weights ∈ {{-1,0,1}}");
        let zeros = (w1q.iter().chain(&w2q).filter(|&&v| v == 0.0).count()) as f64 / (w1q.len() + w2q.len()) as f64;
        assert!(zeros > 0.15, "sparse (multiply-free)");

        // still generalizes clearly above chance (honest quantization tax)
        assert!(te > 0.78, "ternary net generalizes: test {:.2}", te);
    }
}
