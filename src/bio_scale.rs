//! bio_scale (learning #7, de-risk): local DFA learning GENERALIZES on held-out data (3×3 checkerboard),
//! beating a linear baseline — no backprop. Reuses the DFA net from bio_net. Ported from bio_scale.py.

use crate::bio_net::BioNet;
use crate::rng::Rng;
use crate::vec;

pub fn make_checkerboard(n: usize, rng: &mut Rng) -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let (a, b) = (rng.uniform(), rng.uniform());
        let lab = (((a * 3.0).floor() + (b * 3.0).floor()) as i64 % 2) as f64;
        x.push([a * 2.0 - 1.0, b * 2.0 - 1.0]); // centered to [-1,1]
        y.push(lab);
    }
    (x, y)
}

/// linear baseline (no hidden layer)
struct Linear {
    w: [f64; 2],
    b: f64,
    lr: f64,
    x: [f64; 2],
    yy: f64,
}
impl Linear {
    fn new(lr: f64) -> Self {
        Linear { w: [0.0; 2], b: 0.0, lr, x: [0.0; 2], yy: 0.0 }
    }
    fn forward(&mut self, x: &[f64; 2]) -> f64 {
        self.x = *x;
        self.yy = vec::sigmoid(self.w[0] * x[0] + self.w[1] * x[1] + self.b);
        self.yy
    }
    fn learn(&mut self, t: f64) {
        let e = self.yy - t;
        self.w[0] -= self.lr * e * self.x[0];
        self.w[1] -= self.lr * e * self.x[1];
        self.b -= self.lr * e;
    }
}

fn eval_net(net: &mut BioNet, x: &[[f64; 2]], y: &[f64]) -> f64 {
    let c = (0..x.len()).filter(|&i| (net.forward(&x[i])[0] > 0.5) == (y[i] > 0.5)).count();
    c as f64 / x.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dfa_generalizes_beats_linear() {
        let mut g = Rng::new(0);
        let (xtr, ytr) = make_checkerboard(1000, &mut g);
        let (xte, yte) = make_checkerboard(500, &mut g);
        let mut net = BioNet::new(2, 128, 1, 0.15, &mut g);
        let mut lin = Linear::new(0.2);
        for _ in 0..600 {
            for &i in &g.randperm(xtr.len()) {
                net.forward(&xtr[i]);
                net.learn(&[ytr[i]]);
                lin.forward(&xtr[i]);
                lin.learn(ytr[i]);
            }
        }
        let bio_te = eval_net(&mut net, &xte, &yte);
        let lin_te = (0..xte.len()).filter(|&i| (lin.forward(&xte[i]) > 0.5) == (yte[i] > 0.5)).count() as f64 / xte.len() as f64;
        assert!(bio_te > 0.90 && bio_te - 0.15 > lin_te, "DFA generalizes (test {:.2}) beating linear ({:.2})", bio_te, lin_te);
    }
}
