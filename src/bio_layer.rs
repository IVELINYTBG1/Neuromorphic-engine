//! bio_layer — the keystone cell (learning principles #1-4,#6): LIF membrane (leak/integrate/reset) +
//! spike-frequency adaptation + k-WTA sparsity + THREE-FACTOR local learning (eligibility × neuromodulator,
//! no backprop) + the neuromodulator threshold warp. Ported from bio_layer.py. Torch replaced by crate::{rng,vec}.

use crate::rng::Rng;
use crate::vec;

pub struct BioLayer {
    pub n_in: usize,
    pub n_out: usize,
    pub k: usize,
    beta: f64,
    theta0: f64,
    adapt: f64,
    adapt_decay: f64,
    lr: f64,
    trace_decay: f64,
    pub w: Vec<f64>, // n_in × n_out row-major — synapses = memory + compute
    pub v: Vec<f64>,
    pub thr_adapt: Vec<f64>,
    pre_tr: Vec<f64>,
    post_tr: Vec<f64>,
    elig: Vec<f64>,
    pub last_drive: Vec<f64>,
}

impl BioLayer {
    pub fn new(n_in: usize, n_out: usize, k: usize, rng: &mut Rng) -> Self {
        let w = rng.randn_vec(n_in * n_out, 0.05);
        let mut l = BioLayer {
            n_in, n_out, k,
            beta: 0.9, theta0: 0.4, adapt: 0.05, adapt_decay: 0.9, lr: 0.05, trace_decay: 0.6,
            w,
            v: vec![], thr_adapt: vec![], pre_tr: vec![], post_tr: vec![], elig: vec![], last_drive: vec![],
        };
        l.reset_state();
        l
    }

    pub fn reset_state(&mut self) {
        self.v = vec![0.0; self.n_out];
        self.thr_adapt = vec![0.0; self.n_out];
        self.pre_tr = vec![0.0; self.n_in];
        self.post_tr = vec![0.0; self.n_out];
        self.elig = vec![0.0; self.n_in * self.n_out];
        self.last_drive = vec![0.0; self.n_out];
    }

    /// One tick. x: presyn spikes; target: optional taught post (clamp); the neuromodulator warps θ.
    pub fn forward(&mut self, x: &[f64], target: Option<&[f64]>, v_phill: f64, alpha: f64) -> Vec<f64> {
        let i_cur = vec::matvec(x, &self.w, self.n_in, self.n_out); // synaptic current
        let mut theta = vec![0.0; self.n_out];
        let mut drive = vec![0.0; self.n_out];
        for j in 0..self.n_out {
            self.v[j] = self.beta * self.v[j] + i_cur[j]; // LIF: leak + integrate
            theta[j] = self.theta0 + self.thr_adapt[j] + alpha * v_phill; // adaptive + the neuromodulator warp
            drive[j] = self.v[j] - theta[j];
        }
        self.last_drive = drive.clone();

        // k-WTA sparsity: top-k neurons above threshold fire
        let mut s = vec![0.0; self.n_out];
        let cand: Vec<usize> = (0..self.n_out).filter(|&j| drive[j] > 0.0).collect();
        if !cand.is_empty() {
            let mut c = cand.clone();
            c.sort_by(|&a, &b| drive[b].partial_cmp(&drive[a]).unwrap());
            for &j in c.iter().take(self.k) {
                s[j] = 1.0;
            }
        }
        let post: Vec<f64> = match target { Some(t) => t.to_vec(), None => s.clone() };

        for j in 0..self.n_out {
            self.v[j] -= post[j] * theta[j]; // reset whoever fired
            self.thr_adapt[j] = self.adapt_decay * self.thr_adapt[j] + self.adapt * post[j]; // adaptation
        }
        for i in 0..self.n_in {
            self.pre_tr[i] = self.trace_decay * self.pre_tr[i] + x[i];
        }
        for j in 0..self.n_out {
            self.post_tr[j] = self.trace_decay * self.post_tr[j] + post[j];
        }
        vec::add_outer(&mut self.elig, &self.pre_tr, &post, 1.0); // eligibility = outer(pre_tr, post)
        s
    }

    /// three-factor rule: tagged eligibility → weight change, GATED by M (dopamine / the neuromodulator salience)
    pub fn neuromodulate(&mut self, m: f64) {
        for idx in 0..self.w.len() {
            self.w[idx] = vec::clamp(self.w[idx] + self.lr * m * self.elig[idx], -2.0, 2.0);
            self.elig[idx] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterns() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut x = vec![vec![0.0; 8]; 4];
        for i in 0..4 {
            x[i][2 * i] = 1.0;
            x[i][2 * i + 1] = 1.0;
        }
        let mut y = vec![vec![0.0; 4]; 4];
        for i in 0..4 {
            y[i][i] = 1.0;
        }
        (x, y)
    }

    fn accuracy(l: &mut BioLayer, x: &[Vec<f64>]) -> f64 {
        let mut c = 0;
        for i in 0..4 {
            l.reset_state();
            l.forward(&x[i], None, 0.0, 0.0);
            if vec::argmax(&l.last_drive) == i {
                c += 1;
            }
        }
        c as f64 / 4.0
    }

    #[test]
    fn learns_locally_no_backprop() {
        let (x, y) = patterns();
        let mut rng = Rng::new(0);
        let mut l = BioLayer::new(8, 4, 1, &mut rng);

        assert!(accuracy(&mut l, &x) <= 0.5); // starts poor
        for _ in 0..25 {
            for i in 0..4 {
                l.reset_state();
                l.forward(&x[i], Some(&y[i]), 0.0, 0.0);
                l.neuromodulate(1.0);
            }
        }
        let acc = accuracy(&mut l, &x);
        assert_eq!(acc, 1.0, "three-factor local learning should reach 100%");

        // k-WTA sparsity
        l.reset_state();
        let s = l.forward(&x[0], None, 0.0, 0.0);
        assert_eq!(vec::sum(&s) as i32, 1, "k-WTA k=1 → exactly one fires");

        // the neuromodulator warp inhibits (an agent under feeling)
        l.reset_state();
        l.forward(&x[0], None, 0.0, 0.0);
        let calm = vec::max(&l.last_drive);
        l.reset_state();
        l.forward(&x[0], None, 0.9, 0.40);
        let warp = vec::max(&l.last_drive);
        assert!(warp < calm, "the neuromodulator warp should inhibit ({} !< {})", warp, calm);

        // #1 membrane: integrate then leak
        l.reset_state();
        for _ in 0..3 {
            l.forward(&vec![0.2; 8], None, 0.0, 0.0);
        }
        let charged: f64 = l.v.iter().map(|z| z.abs()).sum();
        for _ in 0..10 {
            l.forward(&vec![0.0; 8], None, 0.0, 0.0);
        }
        let leaked: f64 = l.v.iter().map(|z| z.abs()).sum();
        assert!(leaked < charged + 1e-6, "membrane should leak");

        // #2 adaptation
        l.reset_state();
        for _ in 0..5 {
            l.forward(&x[0], None, 0.0, 0.0);
        }
        assert!(vec::max(&l.thr_adapt) > 0.0, "adaptive threshold should rise");
    }
}
