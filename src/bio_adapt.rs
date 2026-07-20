//! bio_adapt (motor #5): two-state motor adaptation (Smith/Shadmehr) — recalibrates a feedforward model
//! from error: adapts, leaves an aftereffect (opposite sign), and a fast+slow memory gives savings even
//! after full washout (a single state shows none). Ported from bio_adapt.py.

pub fn run_protocol(schedule: &[f64], a_f: f64, b_f: f64, a_s: f64, b_s: f64, two_state: bool) -> Vec<f64> {
    let (mut xf, mut xs) = (0.0, 0.0);
    let mut errs = vec![0.0; schedule.len()];
    for (n, &p) in schedule.iter().enumerate() {
        let x = if two_state { xf + xs } else { xf };
        let e = p - x;
        errs[n] = e;
        xf = a_f * xf + b_f * e;
        if two_state {
            xs = a_s * xs + b_s * e;
        }
    }
    errs
}

fn abs_mean(e: &[f64]) -> f64 {
    e.iter().map(|x| x.abs()).sum::<f64>() / e.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptation_aftereffect_savings() {
        let (base, learn, wash) = (20usize, 120usize, 50usize);
        let mut schedule = vec![];
        schedule.extend(vec![0.0; base]);
        schedule.extend(vec![1.0; learn]);
        schedule.extend(vec![0.0; wash]);
        schedule.extend(vec![1.0; learn]);
        let (learn1, wash0, learn2) = (base, base + learn, base + learn + wash);

        let errs = run_protocol(&schedule, 0.6, 0.35, 0.998, 0.02, true);

        // adapts: residual error near 0 by end of block 1
        assert!(abs_mean(&errs[learn1 + learn - 20..learn1 + learn]) < 0.2, "adapts");

        // aftereffect: first washout reach errs the OPPOSITE way (a model, not online feedback)
        let after = errs[wash0];
        let learn_sign = errs[learn1];
        assert!(after < -0.5 && after * learn_sign < 0.0, "aftereffect");

        // savings: re-learning is faster (smaller early-block error)
        let early = |s: usize| abs_mean(&errs[s..s + 15]);
        assert!(early(learn2) < 0.85 * early(learn1), "savings");

        // savings needs the slow state: a single-state learner (fully washed out) shows none
        let errs1 = run_protocol(&schedule, 0.95, 0.3, 0.0, 0.0, false);
        let early1 = |s: usize| abs_mean(&errs1[s..s + 15]);
        let single_adapts = abs_mean(&errs1[learn1 + learn - 20..learn1 + learn]) < 0.2;
        assert!(early1(learn2) >= 0.98 * early1(learn1) && single_adapts, "single-state: no savings");
    }
}
