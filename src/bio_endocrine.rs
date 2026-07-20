//! bio_endocrine (endocrine #1): sex hormones as a SLOW neuromodulator (gonadal axis). Organizational
//! baseline (trait) + activational cyclic level (state); androgen→approach bias, estradiol→plasticity,
//! HPG feedback; continuous, non-stereotyped (population distributions overlap). From bio_endocrine.py.

use crate::rng::Rng;

pub struct GonadalAxis {
    a0: f64,
    e0: f64,
    cycle: f64,
    k_approach: f64,
    k_plasticity: f64,
    feedback: f64,
    pub a: f64,
    pub e: f64,
    t: f64,
}

impl GonadalAxis {
    pub fn new(androgen: f64, estrogen: f64) -> Self {
        GonadalAxis { a0: androgen, e0: estrogen, cycle: 240.0, k_approach: 1.6, k_plasticity: 1.2,
                      feedback: 0.06, a: androgen, e: estrogen, t: 0.0 }
    }
    pub fn tick(&mut self, perturb_e: f64, activational: bool) {
        let cyc = 0.5 * (1.0 + (2.0 * std::f64::consts::PI * self.t / self.cycle).sin());
        let target_e = if activational { self.e0 * (0.4 + 0.6 * cyc) } else { self.e0 };
        self.a += self.feedback * (self.a0 - self.a);
        self.e += self.feedback * (target_e - self.e) + perturb_e;
        self.t += 1.0;
    }
    pub fn approach_bias(&self) -> f64 {
        (self.k_approach * self.a).tanh()
    }
    pub fn plasticity(&self) -> f64 {
        0.1 * (1.0 + self.k_plasticity * self.e.max(0.0))
    }
    pub fn decide_approach(&self, reward: f64, threat: f64) -> f64 {
        let net = reward - threat + self.approach_bias();
        1.0 / (1.0 + (-4.0 * net).exp())
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sex_hormones_as_slow_modulator() {
        // slow & tonic: estrogen cycles slowly (small per-tick step, ~6 sign changes over 3 cycles)
        let mut ax = GonadalAxis::new(0.0, 1.0);
        let mut e_trace = vec![];
        for _ in 0..720 {
            ax.tick(0.0, true);
            e_trace.push(ax.e);
        }
        let rng_span = e_trace.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - e_trace.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_step = (1..e_trace.len()).map(|i| (e_trace[i] - e_trace[i - 1]).abs()).fold(0.0, f64::max);
        let mean = e_trace.iter().sum::<f64>() / e_trace.len() as f64;
        let crossings = (1..e_trace.len())
            .filter(|&i| (e_trace[i] > mean) != (e_trace[i - 1] > mean))
            .count();
        assert!(max_step < 0.05 * rng_span && (4..=8).contains(&crossings), "slow tonic cycle");

        // androgen tone → approach temperament
        assert!(GonadalAxis::new(0.6, 1.0).decide_approach(0.5, 0.5) > 0.8);
        assert!(GonadalAxis::new(-0.6, 1.0).decide_approach(0.5, 0.5) < 0.2);

        // estradiol gates plasticity over the cycle
        let mut ax2 = GonadalAxis::new(0.0, 1.0);
        let mut plas = vec![];
        for _ in 0..480 {
            ax2.tick(0.0, true);
            plas.push(ax2.plasticity());
        }
        let pmax = plas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pmin = plas.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(pmax > 1.4 * pmin, "cyclic plasticity");

        // organizational (trait) persists with the cyclic state frozen
        let mut trait_only = GonadalAxis::new(0.6, 1.0);
        for _ in 0..50 { trait_only.tick(0.0, false); }
        assert!(trait_only.approach_bias() > 0.5);

        // HPG negative feedback restores the set-point after a surge
        let mut reg = GonadalAxis::new(0.0, 1.0);
        for _ in 0..20 { reg.tick(0.0, false); }
        let settled = reg.e;
        reg.tick(1.5, false);
        let surged = reg.e;
        for _ in 0..150 { reg.tick(0.0, false); }
        assert!(surged > settled + 1.0 && (reg.e - settled).abs() < 0.1, "HPG feedback restores set-point");

        // continuity/overlap: small group-mean gap + big variation → distributions overlap
        let mut r = Rng::new(0);
        let mut population = |mean: f64, r: &mut Rng| -> Vec<f64> {
            (0..400).map(|_| GonadalAxis::new(mean + 0.5 * r.normal(), 1.0).decide_approach(0.5, 0.5)).collect()
        };
        let grp_x = population(0.25, &mut r);
        let grp_y = population(-0.25, &mut r);
        let med_x = median(grp_x);
        let overlap = grp_y.iter().filter(|&&p| p > med_x).count() as f64 / grp_y.len() as f64;
        assert!(overlap > 0.1, "distributions overlap (continuous, not binary)");
    }
}
