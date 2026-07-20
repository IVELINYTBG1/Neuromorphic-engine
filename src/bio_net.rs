//! bio_net (learning #7): cross-layer credit assignment by Direct Feedback Alignment — a FIXED random
//! feedback matrix, no weight transport, no backprop. Solves XOR. Ported from bio_net.py.

use crate::rng::Rng;
use crate::vec;

fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub struct BioNet {
    n_in: usize,
    n_hid: usize,
    n_out: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    bfb: Vec<f64>, // fixed random feedback (n_hid × n_out) — the DFA core
    lr: f64,
    x: Vec<f64>,
    h: Vec<f64>,
    y: Vec<f64>,
}

impl BioNet {
    pub fn new(n_in: usize, n_hid: usize, n_out: usize, lr: f64, rng: &mut Rng) -> Self {
        BioNet {
            n_in, n_hid, n_out,
            w1: rng.randn_vec(n_hid * n_in, 0.5),
            b1: vec![0.0; n_hid],
            w2: rng.randn_vec(n_out * n_hid, 0.5),
            b2: vec![0.0; n_out],
            bfb: rng.randn_vec(n_hid * n_out, 1.0),
            lr,
            x: vec![], h: vec![], y: vec![],
        }
    }
    pub fn forward(&mut self, x: &[f64]) -> Vec<f64> {
        self.x = x.to_vec();
        let a1 = vec::mv(&self.w1, x, self.n_hid, self.n_in);
        self.h = (0..self.n_hid).map(|i| tanh(a1[i] + self.b1[i])).collect();
        let a2 = vec::mv(&self.w2, &self.h, self.n_out, self.n_hid);
        self.y = (0..self.n_out).map(|i| vec::sigmoid(a2[i] + self.b2[i])).collect();
        self.y.clone()
    }
    pub fn learn(&mut self, target: &[f64]) {
        let e: Vec<f64> = (0..self.n_out).map(|i| self.y[i] - target[i]).collect();
        vec::add_outer(&mut self.w2, &e, &self.h, -self.lr);
        for i in 0..self.n_out {
            self.b2[i] -= self.lr * e[i];
        }
        let be = vec::mv(&self.bfb, &e, self.n_hid, self.n_out); // B @ e (fixed feedback)
        let delta_h: Vec<f64> = (0..self.n_hid).map(|i| be[i] * (1.0 - self.h[i] * self.h[i])).collect();
        vec::add_outer(&mut self.w1, &delta_h, &self.x, -self.lr);
        for i in 0..self.n_hid {
            self.b1[i] -= self.lr * delta_h[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_without_backprop() {
        let x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = [[0.0], [1.0], [1.0], [0.0]];
        let mut rng = Rng::new(0);
        let mut net = BioNet::new(2, 16, 1, 0.3, &mut rng);
        for _ in 0..4000 {
            for &i in &rng.randperm(4) {
                net.forward(&x[i]);
                net.learn(&y[i]);
            }
        }
        let acc = (0..4).filter(|&i| (net.forward(&x[i])[0] > 0.5) == (y[i][0] > 0.5)).count();
        assert_eq!(acc, 4, "DFA solves XOR (cross-layer credit, no backprop)");
    }
}
