//! bio_interoception (sensing #7): the insula — the body sensing itself. Feeling constructed from
//! interoceptive prediction error (arousal) + homeostatic deviation (valence) + context (two-factor).
//! Body = 4 channels [heart_rate, respiration, glucose, temperature]. Ported from bio_interoception.py.

use crate::bio_affect::CoreAffect;

pub const SETPOINTS: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
pub const CALM: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
pub const AROUSED: [f64; 4] = [2.0, 1.8, 1.0, 1.1];
pub const HUNGRY: [f64; 4] = [1.1, 1.0, 0.1, 1.0];

fn mean_abs(v: &[f64; 4]) -> f64 {
    v.iter().map(|x| x.abs()).sum::<f64>() / 4.0
}

pub struct Insula {
    set: [f64; 4],
    pred: [f64; 4],
    gain: f64,
    learn: f64,
}

impl Insula {
    pub fn new(setpoints: [f64; 4], gain: f64) -> Self {
        Insula { set: setpoints, pred: setpoints, gain, learn: 0.2 }
    }
    pub fn predict(&mut self, body: &[f64; 4]) {
        for k in 0..4 {
            self.pred[k] += self.learn * (body[k] - self.pred[k]);
        }
    }
    /// Returns (valence, arousal). arousal = interoceptive prediction error × gain; valence =
    /// -homeostatic deviation + context.
    pub fn feel(&self, body: &[f64; 4], context: f64) -> (f64, f64) {
        let mut pe = [0.0; 4];
        let mut dev = [0.0; 4];
        for k in 0..4 {
            pe[k] = body[k] - self.pred[k];
            dev[k] = body[k] - self.set[k];
        }
        let arousal = self.gain * mean_abs(&pe);
        let valence = (-mean_abs(&dev) + context).clamp(-1.0, 1.0);
        (valence, arousal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feeling_from_the_body() {
        // body → feeling (James–Lange)
        let a_calm = Insula::new(SETPOINTS, 1.0).feel(&CALM, 0.0).1;
        let a_high = Insula::new(SETPOINTS, 1.0).feel(&AROUSED, 0.0).1;
        assert!(a_calm < 0.05 && a_high > a_calm + 0.3);

        // interoceptive inference: a predicted body attenuates
        let mut ins = Insula::new(SETPOINTS, 1.0);
        let a_first = ins.feel(&AROUSED, 0.0).1;
        for _ in 0..30 { ins.predict(&AROUSED); }
        assert!(ins.feel(&AROUSED, 0.0).1 < 0.4 * a_first);

        // valence from homeostatic deviation
        let v_sated = Insula::new(SETPOINTS, 1.0).feel(&CALM, 0.0).0;
        let v_hungry = Insula::new(SETPOINTS, 1.0).feel(&HUNGRY, 0.0).0;
        assert!(v_sated > -0.05 && v_hungry < v_sated - 0.2);

        // two-factor: same arousal, context flips valence
        let (v_fear, a_fear) = Insula::new(SETPOINTS, 1.0).feel(&AROUSED, -0.8);
        let (v_exc, a_exc) = Insula::new(SETPOINTS, 1.0).feel(&AROUSED, 0.8);
        assert!((a_fear - a_exc).abs() < 1e-9 && v_fear < 0.0 && v_exc > 0.0);

        // deafen the insula → feeling flat (feeling IS interoception)
        assert!(Insula::new(SETPOINTS, 0.0).feel(&AROUSED, 0.0).1 < 0.05 && a_high > 0.3);

        // bridge: interoception is the SOURCE of core affect (grounds bio_affect)
        let mut ca = CoreAffect::default();
        let body_ins = Insula::new(SETPOINTS, 1.0);
        for _ in 0..12 {
            let (v, a) = body_ins.feel(&AROUSED, -0.5);
            ca.event(v, a);
            ca.tick();
        }
        assert!(ca.val_e < 0.0, "interoception drives core affect negative");
    }
}
