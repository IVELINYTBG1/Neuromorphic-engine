//! bio_seq (learning #5, sequences): a recurrent LIF population wired ONLY by population-pair STDP +
//! homeostatic decay LEARNS a sequence — teach "think", cue 't' replays "think". Ported from bio_seq.py.

use crate::vec;

pub struct BioSequenceLayer {
    n: usize,
    theta: f64,
    w_max: f64,
    a_plus: f64,
    a_minus: f64,
    td: f64,
    forget: f64,
    w: Vec<f64>, // n×n, W[i,j] = link i→j
    pre_tr: Vec<f64>,
    post_tr: Vec<f64>,
}

impl BioSequenceLayer {
    pub fn new(n: usize) -> Self {
        BioSequenceLayer { n, theta: 0.5, w_max: 1.5, a_plus: 0.25, a_minus: 0.27, td: 0.4, forget: 0.10,
                           w: vec![0.0; n * n], pre_tr: vec![0.0; n], post_tr: vec![0.0; n] }
    }
    fn reset_state(&mut self) {
        self.pre_tr = vec![0.0; self.n];
        self.post_tr = vec![0.0; self.n];
    }
    fn stdp(&mut self, s: &[f64]) {
        for i in 0..self.n {
            self.pre_tr[i] *= self.td;
            self.post_tr[i] *= self.td;
        }
        vec::add_outer(&mut self.w, &self.pre_tr, s, self.a_plus); // pre→post LTP
        vec::add_outer(&mut self.w, s, &self.post_tr, -self.a_minus); // post→pre LTD
        for i in 0..self.n {
            self.pre_tr[i] += s[i];
            self.post_tr[i] += s[i];
        }
        for i in 0..self.n {
            for j in 0..self.n {
                self.w[i * self.n + j] = self.w[i * self.n + j].clamp(0.0, self.w_max);
            }
            self.w[i * self.n + i] = 0.0;
        }
    }
    pub fn teach(&mut self, seq: &[usize], repeats: usize) {
        for _ in 0..repeats {
            self.reset_state();
            for &sym in seq {
                let mut s = vec![0.0; self.n];
                s[sym] = 1.0;
                self.stdp(&s);
            }
            self.w.iter_mut().for_each(|w| *w *= 1.0 - self.forget); // homeostatic decay
        }
    }
    pub fn recall(&self, cue: usize, steps: usize) -> Vec<usize> {
        let mut s = vec![0.0; self.n];
        s[cue] = 1.0;
        let mut fired = vec![cue];
        for _ in 0..steps {
            let drive = vec::matvec(&s, &self.w, self.n, self.n); // Wᵀ @ s = incoming to each j
            let nxt = vec::argmax(&drive);
            if drive[nxt] <= self.theta {
                break;
            }
            fired.push(nxt);
            s = vec![0.0; self.n];
            s[nxt] = 1.0;
        }
        fired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdp_learns_a_sequence() {
        let word = "think";
        let mut alpha: Vec<char> = word.chars().collect();
        alpha.sort();
        alpha.dedup();
        let seq: Vec<usize> = word.chars().map(|c| alpha.iter().position(|&a| a == c).unwrap()).collect();

        let mut sl = BioSequenceLayer::new(alpha.len());
        assert_eq!(sl.recall(seq[0], seq.len()).len(), 1, "before teaching: nothing fires");
        sl.teach(&seq, 40);
        let out = sl.recall(seq[0], seq.len());
        let replayed: String = out.iter().map(|&i| alpha[i]).collect();
        assert_eq!(replayed, word, "cue replays the taught sequence");
    }
}
