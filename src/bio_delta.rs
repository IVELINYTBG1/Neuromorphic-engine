//! bio_delta (learning #9): event/delta coding — recompute only the input dims that changed. On a sparse-
//! changing (embodiment-like) stream: big compute saving, output matches the dense recompute. From bio_delta.py.

use crate::rng::Rng;
use crate::vec;

pub fn embodiment_stream(t: usize, n_in: usize, changes_per_tick: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut x = vec![vec![0.0; n_in]; t];
    let mut state: Vec<f64> = rng.rand_vec(n_in);
    for row in x.iter_mut() {
        let perm = rng.randperm(n_in);
        for &idx in perm.iter().take(changes_per_tick) {
            state[idx] += rng.normal() * 0.1;
        }
        *row = state.clone();
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_coding_saves_compute() {
        let (t, n_in, n_out) = (300usize, 64usize, 32usize);
        let mut rng = Rng::new(0);
        let w = rng.randn_vec(n_out * n_in, 0.1); // n_out × n_in
        let x = embodiment_stream(t, n_in, 4, &mut rng);
        let eps = 0.01;

        // dense: full matvec every tick
        let mut dense_macs = 0usize;
        let mut yd = vec![vec![0.0; n_out]; t];
        for step in 0..t {
            yd[step] = vec::mv(&w, &x[step], n_out, n_in);
            dense_macs += n_in * n_out;
        }

        // delta: incremental update over changed dims only
        let mut delta_macs = 0usize;
        let mut ye = vec![vec![0.0; n_out]; t];
        let mut x_prev = vec![0.0; n_in];
        let mut y = vec![0.0; n_out];
        for step in 0..t {
            let events: Vec<usize> = (0..n_in).filter(|&j| (x[step][j] - x_prev[j]).abs() > eps).collect();
            for &j in &events {
                let dxj = x[step][j] - x_prev[j];
                for i in 0..n_out {
                    y[i] += w[i * n_in + j] * dxj;
                }
                x_prev[j] = x[step][j];
                delta_macs += n_out;
            }
            ye[step] = y.clone();
        }

        let saved = 1.0 - delta_macs as f64 / dense_macs as f64;
        let match_err = (0..t).flat_map(|s| (0..n_out).map(move |i| (s, i)))
            .map(|(s, i)| (yd[s][i] - ye[s][i]).abs())
            .fold(0.0, f64::max);
        assert!(saved > 0.5 && match_err < 0.05, "delta coding: {:.0}% saved, err {:.4}", saved * 100.0, match_err);
    }
}
