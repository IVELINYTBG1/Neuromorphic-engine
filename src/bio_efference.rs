//! bio_efference (sensing #6): efference copy + forward model — cancels self-caused sensation (can't
//! tickle yourself), lets the external world through; same touch felt only when another causes it.
//! Forward model learned by a local rule (residual × action, no backprop). From bio_efference.py.

use crate::rng::Rng;
use crate::vec;

pub struct EfferenceCopy {
    m: Vec<f64>, // sensory_dim × action_dim (learned forward model)
    a_dim: usize,
    s_dim: usize,
    lr: f64,
}

impl EfferenceCopy {
    pub fn new(action_dim: usize, sensory_dim: usize) -> Self {
        EfferenceCopy { m: vec![0.0; sensory_dim * action_dim], a_dim: action_dim, s_dim: sensory_dim, lr: 0.05 }
    }
    pub fn predict(&self, action: &[f64]) -> Vec<f64> {
        vec::mv(&self.m, action, self.s_dim, self.a_dim)
    }
    pub fn perceive(&self, sensory_in: &[f64], efference_copy: Option<&[f64]>) -> Vec<f64> {
        match efference_copy {
            None => sensory_in.to_vec(),
            Some(ec) => {
                let p = self.predict(ec);
                (0..self.s_dim).map(|i| sensory_in[i] - p[i]).collect()
            }
        }
    }
    pub fn learn(&mut self, action: &[f64], sensory_in: &[f64]) -> f64 {
        let p = self.predict(action);
        let residual: Vec<f64> = (0..self.s_dim).map(|i| sensory_in[i] - p[i]).collect();
        vec::add_outer(&mut self.m, &residual, action, self.lr);
        vec::norm(&residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        vec::dot(a, b) / (vec::norm(a) * vec::norm(b) + 1e-12)
    }

    #[test]
    fn cancels_self_lets_world_through() {
        let (a_dim, s_dim) = (6, 10);
        let mut g = Rng::new(0);
        let body = g.randn_vec(s_dim * a_dim, 0.5); // true action→sensation physics (s_dim × a_dim)

        let mut ec = EfferenceCopy::new(a_dim, s_dim);
        let mut err_first = None;
        let mut err_last = 0.0;
        for _ in 0..600 {
            let u = g.randn_vec(a_dim, 1.0);
            let s = vec::mv(&body, &u, s_dim, a_dim);
            let e = ec.learn(&u, &s);
            if err_first.is_none() {
                err_first = Some(e);
            }
            err_last = e;
        }
        assert!(err_last < 0.1 * err_first.unwrap(), "forward model learned locally");

        // self-caused sensation is cancelled
        let u = g.randn_vec(a_dim, 1.0);
        let s_self = vec::mv(&body, &u, s_dim, a_dim);
        let perceived_self = ec.perceive(&s_self, Some(&u));
        assert!(vec::norm(&perceived_self) < 0.05 * vec::norm(&s_self), "can't tickle yourself");

        // an external stimulus (added to your own action) passes through
        let ext = g.randn_vec(s_dim, 1.0);
        let s_mixed: Vec<f64> = (0..s_dim).map(|i| s_self[i] + ext[i]).collect();
        let perceived_ext = ec.perceive(&s_mixed, Some(&u));
        assert!(cosine(&perceived_ext, &ext) > 0.95, "external part recovered");

        // self vs other dissociation: same stimulus, cancelled when self-caused, felt when other-caused
        let with_ec = vec::norm(&ec.perceive(&s_self, Some(&u)));
        let without_ec = vec::norm(&ec.perceive(&s_self, None));
        assert!(with_ec < 0.1 * without_ec, "self vs other dissociation");
    }
}
