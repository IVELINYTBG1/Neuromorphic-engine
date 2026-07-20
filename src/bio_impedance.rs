//! bio_impedance (motor #6): equilibrium-point control (Feldman) — move by setting where the muscle
//! springs balance, never computing torque. Reach by shifting EP, reject load by stiffness, position &
//! impedance independent, same command lands the same place at any mass. Ported from bio_impedance.py.

pub fn reach(ep: f64, k: f64, m: f64, f_ext: f64) -> Vec<f64> {
    let (b, steps, dt) = (1.0, 2000usize, 0.05);
    let (mut x, mut v) = (0.0f64, 0.0f64);
    let mut out = vec![0.0; steps];
    for t in 0..steps {
        let a = (k * (ep - x) - b * v + f_ext) / m;
        v += dt * a;
        x += dt * v;
        out[t] = x;
    }
    out
}

pub fn settled(out: &[f64]) -> f64 {
    let tail = &out[out.len() - 200..];
    tail.iter().sum::<f64>() / tail.len() as f64
}
fn overshoot(out: &[f64], target: f64) -> f64 {
    out.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_point_control() {
        let target = 1.0;
        // reach by shifting the EP (no torque computed)
        assert!((settled(&reach(target, 4.0, 1.0, 0.0)) - target).abs() < 0.02);

        // stiffness rejects a load: deflection ≈ f/K, smaller at higher K
        let d: Vec<f64> = [2.0, 4.0, 8.0].iter().map(|&k| (settled(&reach(target, k, 1.0, 1.0)) - target).abs()).collect();
        assert!(d[0] > d[1] && d[1] > d[2] && d[2] < 0.2, "stiffness rejects load");

        // position and impedance independent: no-load rest position = EP for every K
        let rest: Vec<f64> = [2.0, 4.0, 8.0].iter().map(|&k| settled(&reach(0.5, k, 1.0, 0.0))).collect();
        assert!(rest.iter().all(|&p| (p - 0.5).abs() < 0.02), "EP unmoved by stiffness");

        // no inverse dynamics: mass-invariant endpoint, only the transient differs
        let finals: Vec<f64> = [0.5, 1.0, 4.0].iter().map(|&m| settled(&reach(target, 4.0, m, 0.0))).collect();
        let shoots: Vec<f64> = [0.5, 1.0, 4.0].iter().map(|&m| overshoot(&reach(target, 4.0, m, 0.0), target)).collect();
        assert!(finals.iter().all(|&f| (f - target).abs() < 0.02) && shoots[2] > 2.0 * shoots[0] + 0.02,
                "endpoint mass-invariant, transient differs");
    }
}
