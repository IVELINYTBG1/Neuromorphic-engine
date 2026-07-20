//! bio_receptive (sensing #2): center-surround lateral inhibition (difference-of-Gaussians) — ignores flat
//! fields, fires on edges, Mach bands, ON/OFF opposite sign, whitening. Ported from bio_receptive.py.

use crate::rng::Rng;
use crate::vec;

pub fn dog_kernel() -> Vec<f64> {
    let (sigma_c, sigma_s, radius) = (1.0, 3.0, 9i32);
    let xs: Vec<f64> = (-radius..=radius).map(|i| i as f64).collect();
    let mut c: Vec<f64> = xs.iter().map(|&x| (-x * x / (2.0 * sigma_c * sigma_c)).exp()).collect();
    let mut s: Vec<f64> = xs.iter().map(|&x| (-x * x / (2.0 * sigma_s * sigma_s)).exp()).collect();
    let (cs, ss): (f64, f64) = (c.iter().sum(), s.iter().sum());
    c.iter_mut().for_each(|v| *v /= cs);
    s.iter_mut().for_each(|v| *v /= ss);
    (0..c.len()).map(|i| c[i] - s[i]).collect()
}

/// 1-D convolution (cross-correlation) with reflect padding — the retinal filter.
pub fn retina(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let r = kernel.len() / 2;
    let n = signal.len();
    let mut padded = Vec::with_capacity(n + 2 * r);
    for k in (1..=r).rev() {
        padded.push(signal[k]); // reflect left
    }
    padded.extend_from_slice(signal);
    for k in 1..=r {
        padded.push(signal[n - 1 - k]); // reflect right
    }
    (0..n).map(|i| (0..kernel.len()).map(|j| padded[i + j] * kernel[j]).sum()).collect()
}

pub fn autocorr(v: &[f64], lag: usize) -> f64 {
    let a = &v[..v.len() - lag];
    let b = &v[lag..];
    let (ma, mb) = (vec::mean(a), vec::mean(b));
    let ca: Vec<f64> = a.iter().map(|x| x - ma).collect();
    let cb: Vec<f64> = b.iter().map(|x| x - mb).collect();
    vec::dot(&ca, &cb) / (vec::norm(&ca) * vec::norm(&cb) + 1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(v: &[f64]) -> f64 {
        v.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }

    #[test]
    fn center_surround() {
        let k = dog_kernel();
        // ignores flat fields, fires on edges
        let uniform = vec![0.7; 100];
        let mut step = vec![0.0; 50];
        step.extend(vec![1.0; 50]);
        assert!(max_abs(&retina(&step, &k)) > 20.0 * (max_abs(&retina(&uniform, &k)) + 1e-6), "edge ≫ flat");

        // Mach bands: overshoot on the bright side, undershoot on the dark side of the step
        let resp = retina(&step, &k);
        assert!(resp[50] > resp[80] + 0.05 && resp[49] < resp[20] - 0.05, "Mach bands");

        // whitening: a correlated random walk → decorrelated output by lag 2–3
        let mut g = Rng::new(0);
        let mut walk = vec![0.0; 400];
        let mut acc = 0.0;
        for i in 0..400 {
            acc += g.normal() * 0.1;
            walk[i] = acc;
        }
        let out = retina(&walk, &k);
        assert!(autocorr(&walk, 3) > 0.9 && autocorr(&out, 3).abs() < 0.3, "whitening");

        // ON- vs OFF-centre opposite sign
        let mut bar = vec![0.0; 60];
        for i in 28..32 {
            bar[i] = 1.0;
        }
        let neg_k: Vec<f64> = k.iter().map(|x| -x).collect();
        assert!(retina(&bar, &k)[30] > 0.1 && retina(&bar, &neg_k)[30] < -0.1, "ON/OFF opposite");
    }
}
