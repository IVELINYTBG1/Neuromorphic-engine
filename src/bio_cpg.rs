//! bio_cpg (motor #2): central pattern generator — a half-centre oscillator makes an intrinsic anti-phase
//! rhythm from constant drive. Tonic drive sets VIGOR, a speed command sets FREQUENCY; kill the reciprocal
//! inhibition and the rhythm dies. Ported from bio_cpg.py.

use crate::vec;

pub struct HalfCenter {
    w_inh: f64,
    b_adapt: f64,
    tau: f64,
    tau_a: f64,
    pub dt: f64,
}

impl Default for HalfCenter {
    fn default() -> Self {
        HalfCenter { w_inh: 2.0, b_adapt: 2.2, tau: 1.0, tau_a: 12.0, dt: 0.1 }
    }
}

impl HalfCenter {
    pub fn run(&self, drive: f64, speed: f64, w_inh: Option<f64>) -> (Vec<f64>, Vec<f64>) {
        let steps = 6000;
        let w = w_inh.unwrap_or(self.w_inh);
        let (tau, tau_a) = (self.tau / speed, self.tau_a / speed);
        let (mut xa, mut xb) = (0.0f64, 0.0f64);
        let (mut aa, mut ab) = (0.1f64, 0.0f64);
        let mut ra = vec![0.0; steps];
        let mut rb = vec![0.0; steps];
        for t in 0..steps {
            let (ya, yb) = (xa.max(0.0), xb.max(0.0));
            xa += self.dt * (-xa + drive - w * yb - self.b_adapt * aa) / tau;
            xb += self.dt * (-xb + drive - w * ya - self.b_adapt * ab) / tau;
            aa += self.dt * (-aa + xa.max(0.0)) / tau_a;
            ab += self.dt * (-ab + xb.max(0.0)) / tau_a;
            ra[t] = xa.max(0.0);
            rb[t] = xb.max(0.0);
        }
        (ra, rb)
    }
}

pub fn period(ra: &[f64], rb: &[f64], dt: f64) -> f64 {
    let sign: Vec<bool> = ra.iter().zip(rb).map(|(a, b)| a - b > 0.0).collect();
    let crossings = (1..sign.len()).filter(|&i| sign[i] != sign[i - 1]).count();
    if crossings < 2 {
        f64::INFINITY
    } else {
        2.0 * ra.len() as f64 * dt / crossings as f64
    }
}

fn amp(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tail(v: &[f64]) -> &[f64] {
        &v[v.len() / 2..]
    }

    #[test]
    fn rhythm_from_no_rhythm() {
        let cpg = HalfCenter::default();
        let (ra, rb) = cpg.run(1.4, 1.0, None);
        let per = period(tail(&ra), tail(&rb), cpg.dt);
        assert!(per.is_finite() && (ra.len() / 2) as f64 * cpg.dt / per > 3.0, "sustained rhythm");

        // anti-phase: negative correlation, never co-active
        let corr = vec::pearson(tail(&ra), tail(&rb));
        let coactive = tail(&ra).iter().zip(tail(&rb)).filter(|(a, b)| a.min(**b) > 0.2).count() as f64
            / tail(&ra).len() as f64;
        assert!(corr < -0.3 && coactive < 0.15, "anti-phase gait");

        // drive → vigor (amplitude), cadence fixed
        let lo = cpg.run(0.8, 1.0, None);
        let hi = cpg.run(2.6, 1.0, None);
        let per_lo = period(tail(&lo.0), tail(&lo.1), cpg.dt);
        let per_hi = period(tail(&hi.0), tail(&hi.1), cpg.dt);
        assert!(amp(tail(&hi.0)) > 2.0 * amp(tail(&lo.0)) && (per_hi - per_lo).abs() < 0.15 * per_lo,
                "drive sets vigor, not cadence");

        // speed command → frequency, vigor fixed
        let slow = cpg.run(1.4, 0.7, None);
        let fast = cpg.run(1.4, 2.0, None);
        let f_slow = 1.0 / period(tail(&slow.0), tail(&slow.1), cpg.dt);
        let f_fast = 1.0 / period(tail(&fast.0), tail(&fast.1), cpg.dt);
        assert!(f_fast > 2.0 * f_slow && (amp(tail(&fast.0)) - amp(tail(&slow.0))).abs() < 0.2 * amp(tail(&slow.0)),
                "speed sets cadence, not vigor");

        // rhythm needs reciprocal inhibition
        let (r0a, r0b) = cpg.run(1.4, 1.0, Some(0.0));
        let per0 = period(tail(&r0a), tail(&r0b), cpg.dt);
        assert!(!per0.is_finite() || per0 > 5.0 * per, "no inhibition → rhythm dies");
    }
}
