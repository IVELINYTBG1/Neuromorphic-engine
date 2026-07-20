//! bio_oscillate (thinking #3): rhythms bind (communication-through-coherence) & multiplex (theta–gamma
//! phase code, 7±2 capacity). Ported from bio_oscillate.py.

use crate::rng::Rng;
use std::collections::HashMap;
use std::f64::consts::PI;

fn excitability(phase: f64, kappa: f64) -> f64 {
    (kappa * (phase.cos() - 1.0)).exp()
}

pub fn coherence_transfer(d_phi: f64) -> f64 {
    let (kappa, n) = (4.0, 720);
    let mut acc = 0.0;
    for i in 0..n {
        let phi = 2.0 * PI * i as f64 / (n - 1) as f64;
        acc += excitability(phi, kappa) * excitability(phi + d_phi, kappa);
    }
    acc / n as f64
}

pub struct ThetaGammaBuffer {
    capacity: usize,
}

impl ThetaGammaBuffer {
    pub fn new(capacity: usize) -> Self {
        ThetaGammaBuffer { capacity }
    }
    /// gamma-paced WTA + adaptation → {item: gamma-phase-slot}
    pub fn encode(&self, priorities: &[f64]) -> HashMap<usize, usize> {
        let mut fired = HashMap::new();
        let mut adapted = vec![false; priorities.len()];
        for slot in 0..self.capacity {
            let mut best: Option<usize> = None;
            let mut best_p = f64::NEG_INFINITY;
            for k in 0..priorities.len() {
                if !adapted[k] && priorities[k] > best_p {
                    best_p = priorities[k];
                    best = Some(k);
                }
            }
            match best {
                Some(k) => {
                    fired.insert(k, slot);
                    adapted[k] = true;
                }
                None => break,
            }
        }
        fired
    }
    pub fn decode_order(fired: &HashMap<usize, usize>) -> Vec<usize> {
        let mut v: Vec<(usize, usize)> = fired.iter().map(|(&k, &s)| (k, s)).collect();
        v.sort_by_key(|&(_, s)| s);
        v.into_iter().map(|(k, _)| k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherence_and_phase_code() {
        // communication-through-coherence: in-phase ≫ anti-phase, peak at zero offset
        let sweep: Vec<f64> = [0.0, 0.25, 0.5, 0.75, 1.0].iter().map(|&d| coherence_transfer(d * PI)).collect();
        let in_phase = sweep[0];
        let anti_phase = sweep[4];
        assert!(in_phase / anti_phase > 10.0 && sweep.iter().all(|&t| in_phase >= t), "coherence routes");

        // theta–gamma phase code recovers item order from firing phase
        let mut g = Rng::new(0);
        let k = 5;
        let pri: Vec<f64> = (0..k).map(|_| g.uniform() + 0.1).collect();
        let mut true_order: Vec<usize> = (0..k).collect();
        true_order.sort_by(|&a, &b| pri[b].partial_cmp(&pri[a]).unwrap());
        let buf = ThetaGammaBuffer::new(7);
        let fired = buf.encode(&pri);
        assert!(ThetaGammaBuffer::decode_order(&fired) == true_order && fired.len() == k, "phase code recovers order");

        // 7±2 capacity: 12 offered, 7 held
        let pri_big: Vec<f64> = (0..12).map(|_| g.uniform() + 0.1).collect();
        assert_eq!(buf.encode(&pri_big).len(), 7, "capacity ≈ 7");
    }
}
