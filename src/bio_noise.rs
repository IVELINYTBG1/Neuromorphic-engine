//! bio_noise (learning #5): noise tolerance via distributed a-of-N population codes — graceful degradation
//! + pattern completion, where one-hot is brittle. Local delta-rule readout. Ported from bio_noise.py.

use crate::rng::Rng;
use crate::vec;

pub fn make_distributed_codes(m: usize, n: usize, a: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    (0..m).map(|_| {
        let mut c = vec![0.0; n];
        for &idx in rng.randperm(n).iter().take(a) {
            c[idx] = 1.0;
        }
        c
    }).collect()
}

pub fn corrupt(code: &[f64], p: f64, rng: &mut Rng) -> Vec<f64> {
    code.iter().map(|&c| {
        let flip = if rng.uniform() < p { 1.0 } else { 0.0 };
        c + flip - 2.0 * c * flip // XOR
    }).collect()
}

pub fn train_readout(codes: &[Vec<f64>], succ: &[usize]) -> Vec<f64> {
    let (m, n) = (codes.len(), codes[0].len());
    let mut w = vec![0.0; m * n];
    for _ in 0..300 {
        for s in 0..m {
            let p = vec::softmax(&vec::mv(&w, &codes[s], m, n));
            let mut err = p;
            err[succ[s]] -= 1.0;
            vec::add_outer(&mut w, &err, &codes[s], -0.3);
        }
    }
    w
}

fn predict(w: &[f64], code: &[f64], m: usize, n: usize) -> usize {
    vec::argmax(&vec::mv(w, code, m, n))
}

pub fn noisy_accuracy(w: &[f64], codes: &[Vec<f64>], succ: &[usize], p: f64, trials: usize, rng: &mut Rng) -> f64 {
    let (m, n) = (codes.len(), codes[0].len());
    let (mut correct, mut total) = (0, 0);
    for s in 0..m {
        for _ in 0..trials {
            if predict(w, &corrupt(&codes[s], p, rng), m, n) == succ[s] {
                correct += 1;
            }
            total += 1;
        }
    }
    correct as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributed_codes_are_noise_tolerant() {
        let (m, n, a) = (6, 64, 8);
        let succ: Vec<usize> = (0..m).map(|s| (s + 1) % m).collect();
        let mut g = Rng::new(0);
        let dist = make_distributed_codes(m, n, a, &mut g);
        let oneh: Vec<Vec<f64>> = (0..m).map(|i| { let mut e = vec![0.0; m]; e[i] = 1.0; e }).collect();
        let wd = train_readout(&dist, &succ);
        let wo = train_readout(&oneh, &succ);

        // distributed holds recall under 15% bit-flip where one-hot collapses
        let mut ge = Rng::new(1);
        let accd = noisy_accuracy(&wd, &dist, &succ, 0.15, 400, &mut ge);
        let acco = noisy_accuracy(&wo, &oneh, &succ, 0.15, 400, &mut ge);
        assert!(accd >= 0.90 && accd - acco >= 0.15, "distributed noise-tolerant, beats one-hot");

        // pattern completion: a 20%-corrupted cue still completes the whole sequence
        let mut gc = Rng::new(7);
        let noisy = corrupt(&dist[0], 0.20, &mut gc);
        let mut out = vec![predict(&wd, &noisy, m, n)];
        for _ in 0..m - 1 {
            out.push(predict(&wd, &dist[*out.last().unwrap()], m, n));
        }
        assert_eq!(out, vec![1, 2, 3, 4, 5, 0], "pattern completion from a corrupted cue");
    }
}
