//! bio_fuse (sensing #5): optimal multisensory fusion by ADDING population codes (Ma/Pouget). Reliability-
//! weighted (ventriloquist), precisions add, fused variance ≈ Bayes optimum, beats both cues. From bio_fuse.py.

use crate::rng::Rng;

const XN: usize = 120;
const SIGMA_TC: f64 = 12.0;

pub fn population(mu: f64, gain: f64) -> Vec<f64> {
    (0..XN).map(|i| gain * (-((i as f64 - mu).powi(2)) / (2.0 * SIGMA_TC * SIGMA_TC)).exp()).collect()
}

pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    (0..XN).map(|i| a[i] + b[i]).collect()
}

pub fn decode(r: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..XN {
        num += r[i] * i as f64;
        den += r[i];
    }
    num / (den + 1e-9)
}

fn poisson_pop(mu: f64, gain: f64, rng: &mut Rng) -> Vec<f64> {
    population(mu, gain).iter().map(|&l| rng.poisson(l)).collect()
}

fn decode_var(mu: f64, gain: f64, seed: u64) -> f64 {
    let mut g = Rng::new(seed);
    let ests: Vec<f64> = (0..3000).map(|_| decode(&poisson_pop(mu, gain, &mut g))).collect();
    let m = ests.iter().sum::<f64>() / ests.len() as f64;
    ests.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ests.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_fusion_by_adding_populations() {
        let (mu_v, mu_a) = (40.0, 60.0);
        // ventriloquist: the more reliable cue captures the fused percept
        let fused_vision = decode(&add(&population(mu_v, 4.0), &population(mu_a, 1.0)));
        let fused_audio = decode(&add(&population(mu_v, 1.0), &population(mu_a, 4.0)));
        assert!((fused_vision - mu_v).abs() < (fused_vision - mu_a).abs()
            && (fused_audio - mu_a).abs() < (fused_audio - mu_v).abs(), "reliability-weighted");

        // precisions add when populations are summed
        let (r_v, r_a) = (population(50.0, 3.0), population(50.0, 2.0));
        let (pv, pa): (f64, f64) = (r_v.iter().sum(), r_a.iter().sum());
        let pf: f64 = add(&r_v, &r_a).iter().sum();
        assert!((pf - (pv + pa)).abs() < 1e-3 && pf > pv.max(pa), "precisions add");

        // fused variance under Poisson noise beats both and matches the Bayes optimum
        let (mu, gain_v, gain_a) = (60.0, 20.0, 10.0);
        let var_v = decode_var(mu, gain_v, 0);
        let var_a = decode_var(mu, gain_a, 0);
        let mut g = Rng::new(0);
        let ests: Vec<f64> = (0..3000)
            .map(|_| decode(&add(&poisson_pop(mu, gain_v, &mut g), &poisson_pop(mu, gain_a, &mut g))))
            .collect();
        let m = ests.iter().sum::<f64>() / ests.len() as f64;
        let var_f = ests.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ests.len() as f64;
        let var_opt = (var_v * var_a) / (var_v + var_a);
        assert!(var_f < var_v && var_f < var_a, "fused beats both cues");
        assert!((var_f - var_opt).abs() < 0.25 * var_opt, "fused variance ≈ Bayes optimum");
    }
}
