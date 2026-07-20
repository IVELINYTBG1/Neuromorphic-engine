//! bio_traj (thinking #2): a train of thought = a self-propelled walk across attractors (heteroclinic
//! latching) — symmetric weights make each thought a clean attractor, asymmetric transitions + adaptation
//! move the stream on. Ported from bio_traj.py.

use crate::rng::Rng;
use crate::vec;

pub struct ThoughtTrajectory {
    n: usize,
    lam: f64,
    gamma: f64,
    tau: f64,
    tau_a: f64,
    dt: f64,
    beta: f64,
    w_sym: Vec<f64>,
    w_asym: Vec<f64>,
    patterns: Vec<Vec<f64>>, // K × n
}

impl ThoughtTrajectory {
    pub fn new(n: usize) -> Self {
        ThoughtTrajectory { n, lam: 2.0, gamma: 1.0, tau: 1.0, tau_a: 40.0, dt: 0.1, beta: 8.0,
                            w_sym: vec![], w_asym: vec![], patterns: vec![] }
    }
    pub fn store(&mut self, patterns: Vec<Vec<f64>>) {
        let (n, k) = (self.n, patterns.len());
        let mut w_sym = vec![0.0; n * n];
        for p in &patterns {
            vec::add_outer(&mut w_sym, p, p, 1.0 / n as f64);
        }
        for i in 0..n {
            w_sym[i * n + i] = 0.0;
        }
        let mut w_asym = vec![0.0; n * n];
        for kk in 0..k - 1 {
            vec::add_outer(&mut w_asym, &patterns[kk + 1], &patterns[kk], self.lam / n as f64); // successor ← current
        }
        for i in 0..n {
            w_asym[i * n + i] = 0.0;
        }
        self.w_sym = w_sym;
        self.w_asym = w_asym;
        self.patterns = patterns;
    }
    /// Returns (trajectory of (winner, overlap), peak overlap per pattern).
    pub fn run(&self, cue: &[f64]) -> (Vec<(usize, f64)>, Vec<f64>) {
        let n = self.n;
        let mut u: Vec<f64> = cue.iter().map(|&c| self.beta * c).collect();
        let mut a = vec![0.0; n];
        let mut r: Vec<f64> = u.iter().map(|x| x.tanh()).collect();
        let mut traj = vec![];
        let mut peak = vec![f64::NEG_INFINITY; self.patterns.len()];
        for _ in 0..1400 {
            let sym = vec::mv(&self.w_sym, &r, n, n);
            let asym = vec::mv(&self.w_asym, &a, n, n);
            for i in 0..n {
                let field = sym[i] + asym[i] - self.gamma * a[i];
                u[i] += (self.dt / self.tau) * (-u[i] + field);
                a[i] += (self.dt / self.tau_a) * (-a[i] + r[i].max(0.0));
            }
            for i in 0..n {
                r[i] = (self.beta * u[i]).tanh();
            }
            let mut best_k = 0;
            let mut best_m = f64::NEG_INFINITY;
            for (kk, p) in self.patterns.iter().enumerate() {
                let m = vec::dot(p, &r) / n as f64;
                if m > peak[kk] {
                    peak[kk] = m;
                }
                if m > best_m {
                    best_m = m;
                    best_k = kk;
                }
            }
            traj.push((best_k, best_m));
        }
        (traj, peak)
    }
}

pub fn visiting_order(traj: &[(usize, f64)]) -> Vec<usize> {
    let mut order: Vec<usize> = vec![];
    for &(k, m) in traj {
        if m > 0.5 && order.last() != Some(&k) {
            order.push(k);
        }
    }
    order
}

pub fn first_visits(order: &[usize]) -> Vec<usize> {
    let mut seen = vec![];
    for &k in order {
        if !seen.contains(&k) {
            seen.push(k);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_of_thought() {
        let (n, k) = (100, 5);
        let mut g = Rng::new(0);
        let pats: Vec<Vec<f64>> = (0..k).map(|_| (0..n).map(|_| (g.randint(2) as f64) * 2.0 - 1.0).collect()).collect();
        let mut net = ThoughtTrajectory::new(n);
        net.store(pats.clone());
        let (traj, peak) = net.run(&pats[0]);
        assert_eq!(first_visits(&visiting_order(&traj)), vec![0, 1, 2, 3, 4], "ordered walk");
        assert!(vec::min(&peak) > 0.9, "each thought a clean attractor");
    }
}
