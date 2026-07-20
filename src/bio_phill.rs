//! bio_phill (learning #4 capstone): the neuromodulator as the neuromodulator M — a local delta rule whose plasticity
//! is GATED by felt salience M → emotion-selective memory (salient kept, neutral fades; flashbulb).
//! Ported from bio_phill.py.

use crate::rng::Rng;
use crate::vec;

pub struct SalienceGatedMemory {
    n: usize,
    lr: f64,
    w: Vec<f64>, // n×n
}

impl SalienceGatedMemory {
    pub fn new(n: usize, lr: f64) -> Self {
        SalienceGatedMemory { n, lr, w: vec![0.0; n * n] }
    }
    fn onehot(&self, i: usize) -> Vec<f64> {
        let mut x = vec![0.0; self.n];
        x[i] = 1.0;
        x
    }
    pub fn expose(&mut self, cue: usize, target: usize, m: f64) {
        let x = self.onehot(cue);
        let mut e = vec::softmax(&vec::mv(&self.w, &x, self.n, self.n));
        e[target] -= 1.0;
        vec::add_outer(&mut self.w, &e, &x, -self.lr * m); // feeling gates the change
    }
    pub fn confidence(&self, cue: usize, target: usize) -> f64 {
        vec::softmax(&vec::mv(&self.w, &self.onehot(cue), self.n, self.n))[target]
    }
}

pub fn train(memories: &[(usize, usize)], saliences: &[f64], reps: &[usize], lr: f64, seed: u64)
    -> SalienceGatedMemory {
    let n = 1 + memories.iter().map(|&(c, t)| c.max(t)).max().unwrap();
    let mut mem = SalienceGatedMemory::new(n, lr);
    let mut schedule: Vec<(usize, usize, f64)> = vec![];
    for (k, &(cue, tgt)) in memories.iter().enumerate() {
        for _ in 0..reps[k] {
            schedule.push((cue, tgt, saliences[k]));
        }
    }
    let mut g = Rng::new(seed);
    for j in g.randperm(schedule.len()) {
        let (cue, tgt, m) = schedule[j];
        mem.expose(cue, tgt, m);
    }
    mem
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feeling_gates_memory() {
        let memories = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)];
        let sal = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1];
        let reps = [4usize; 6];

        // equal exposure, unequal feeling → memory tracks salience, not exposure
        let mem = train(&memories, &sal, &reps, 1.5, 0);
        let conf: Vec<f64> = memories.iter().map(|&(c, t)| mem.confidence(c, t)).collect();
        let hi = (conf[0] + conf[1] + conf[2]) / 3.0;
        let lo = (conf[3] + conf[4] + conf[5]) / 3.0;
        assert!(hi > 0.6 && lo < 0.4, "salient kept, neutral fades");

        // flashbulb: ONE salient exposure beats SIX neutral repetitions
        let fb = train(&[(0, 2), (1, 3)], &[1.0, 0.1], &[1, 6], 1.5, 0);
        assert!(fb.confidence(0, 2) > fb.confidence(1, 3), "one salient beats six neutral");

        // M-constant control: everything felt equally → remembered equally (selectivity is M, not data)
        let ctrl = train(&memories, &[0.5; 6], &reps, 1.5, 0);
        let cc: Vec<f64> = memories.iter().map(|&(c, t)| ctrl.confidence(c, t)).collect();
        let spread = vec::max(&cc) - vec::min(&cc);
        assert!(spread < 0.15, "constant M → uniform memory");
    }
}
