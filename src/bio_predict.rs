//! bio_predict (thinking #5): predictive coding (Rao–Ballard) — top-down prediction, bottom-up error,
//! credit by RECIPROCAL connections (no backprop). Inference settles; local Hebbian learning; oddball
//! (surprise ≫ expected); precision = attention gain. Ported from bio_predict.py.

use crate::rng::Rng;
use crate::vec;

pub struct PredictiveCoder {
    d_in: usize,
    d_hidden: usize,
    w: Vec<f64>, // d_in × d_hidden (generative)
    lr_r: f64,
    lr_w: f64,
    decay: f64,
}

impl PredictiveCoder {
    pub fn new(d_in: usize, d_hidden: usize, rng: &mut Rng) -> Self {
        PredictiveCoder { d_in, d_hidden, w: rng.randn_vec(d_in * d_hidden, 0.1),
                          lr_r: 0.1, lr_w: 0.02, decay: 0.05 }
    }
    /// returns (r, error, error-norm curve)
    pub fn infer(&self, x: &[f64], steps: usize, precision: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut r = vec![0.0; self.d_hidden];
        let mut e = x.to_vec();
        let mut errs = vec![];
        for _ in 0..steps {
            let pred = vec::mv(&self.w, &r, self.d_in, self.d_hidden); // W @ r
            e = (0..self.d_in).map(|i| x[i] - pred[i]).collect();
            let wt_e = vec::matvec(&e, &self.w, self.d_in, self.d_hidden); // Wᵀ @ e (reciprocal)
            for j in 0..self.d_hidden {
                r[j] += self.lr_r * (precision * wt_e[j] - self.decay * r[j]);
            }
            errs.push(vec::norm(&e));
        }
        (r, e, errs)
    }
    pub fn learn(&mut self, x_data: &[Vec<f64>], epochs: usize) {
        for _ in 0..epochs {
            for x in x_data {
                let (r, e, _) = self.infer(x, 60, 1.0);
                vec::add_outer(&mut self.w, &e, &r, self.lr_w);
            }
        }
    }
    pub fn residual(&self, x: &[f64], precision: f64) -> f64 {
        vec::norm(&self.infer(x, 120, precision).1)
    }
}

/// Gram–Schmidt orthonormal basis: K columns of length D (replaces torch.linalg.qr).
fn ortho_basis(d: usize, k: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut cols: Vec<Vec<f64>> = (0..k).map(|_| rng.randn_vec(d, 1.0)).collect();
    for i in 0..k {
        for j in 0..i {
            let proj = vec::dot(&cols[i], &cols[j]);
            for d_ in 0..d {
                cols[i][d_] -= proj * cols[j][d_];
            }
        }
        let nrm = vec::norm(&cols[i]) + 1e-12;
        cols[i].iter_mut().for_each(|v| *v /= nrm);
    }
    cols
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictive_coding() {
        let (d, k, h) = (16, 4, 8);
        let mut g = Rng::new(1);
        let basis = ortho_basis(d, k, &mut g); // data live in a K-dim subspace
        let sample = |n: usize, g: &mut Rng| -> Vec<Vec<f64>> {
            (0..n).map(|_| {
                let z = g.randn_vec(k, 1.0);
                let mut x: Vec<f64> = (0..d).map(|di| (0..k).map(|ki| z[ki] * basis[ki][di]).sum()).collect();
                let nrm = vec::norm(&x) + 1e-12;
                x.iter_mut().for_each(|v| *v /= nrm);
                x
            }).collect()
        };
        let x_train = sample(200, &mut g);
        let mut pc = PredictiveCoder::new(d, h, &mut Rng::new(0));

        let err_before = x_train[..40].iter().map(|x| pc.residual(x, 1.0)).sum::<f64>() / 40.0;
        pc.learn(&x_train, 40);
        let err_after = x_train[..40].iter().map(|x| pc.residual(x, 1.0)).sum::<f64>() / 40.0;
        let curve = pc.infer(&x_train[0], 120, 1.0).2;
        assert!(*curve.last().unwrap() < 0.5 * curve[0], "inference settles");
        assert!(err_after < 0.3 * err_before, "local Hebbian learning");

        // oddball: input outside the learned subspace errs far more
        let expected = sample(50, &mut g);
        let perp: Vec<Vec<f64>> = (0..50).map(|_| {
            let v = g.randn_vec(d, 1.0);
            let coeff: Vec<f64> = (0..k).map(|ki| vec::dot(&v, &basis[ki])).collect();
            let mut p: Vec<f64> = (0..d).map(|di| v[di] - (0..k).map(|ki| coeff[ki] * basis[ki][di]).sum::<f64>()).collect();
            let nrm = vec::norm(&p) + 1e-12;
            p.iter_mut().for_each(|x| *x /= nrm);
            p
        }).collect();
        let err_exp = expected.iter().map(|x| pc.residual(x, 1.0)).sum::<f64>() / 50.0;
        let err_sur = perp.iter().map(|x| pc.residual(x, 1.0)).sum::<f64>() / 50.0;
        assert!(err_sur > 2.0 * err_exp, "oddball: surprise ≫ expected");

        // precision = attention gain: higher precision drives the error lower
        assert!(pc.residual(&expected[0], 3.0) < pc.residual(&expected[0], 0.3), "precision attends");
    }
}
