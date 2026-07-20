//! bio_neuroendocrine (endocrine #2): the whole hormone orchestra as a DATA TABLE — 11 hormones × 8 knobs,
//! all modulation a linear map from the level vector through the coupling matrix. Effects/opponency/
//! timescales/circadian are emergent from the numbers; nothing hardcoded. Ported from bio_neuroendocrine.py.

use std::f64::consts::PI;

// knob indices
pub const PLASTICITY: usize = 0;
pub const GAIN: usize = 1;
pub const THREAT_GAIN: usize = 2;
pub const EXPLORATION: usize = 3;
pub const TIME_HORIZON: usize = 4;
pub const SOCIALITY: usize = 5;
pub const VIGOR: usize = 6;
pub const INHIBITION: usize = 7;
pub const N_KNOBS: usize = 8;

#[derive(Clone)]
pub struct Hormone {
    baseline: f64,
    tau: f64,
    couples: Vec<(usize, f64)>,
    circadian: bool,
    feedback: f64,
}

fn default_table() -> Vec<(&'static str, Hormone)> {
    let h = |b, t, c: Vec<(usize, f64)>, cir, fb| Hormone { baseline: b, tau: t, couples: c, circadian: cir, feedback: fb };
    vec![
        ("cortisol", h(0.20, 40.0, vec![(GAIN, 0.5), (THREAT_GAIN, 0.6), (PLASTICITY, -0.4), (TIME_HORIZON, -0.5), (VIGOR, 0.4), (SOCIALITY, -0.3)], false, 0.04)),
        ("oxytocin", h(0.20, 20.0, vec![(SOCIALITY, 0.8), (THREAT_GAIN, -0.5), (PLASTICITY, 0.3)], false, 0.0)),
        ("vasopressin", h(0.20, 25.0, vec![(THREAT_GAIN, 0.4), (VIGOR, 0.3), (SOCIALITY, -0.2)], false, 0.0)),
        ("serotonin", h(0.50, 30.0, vec![(TIME_HORIZON, 0.6), (INHIBITION, 0.5), (THREAT_GAIN, -0.3)], false, 0.0)),
        ("norepinephrine", h(0.30, 10.0, vec![(GAIN, 0.7), (EXPLORATION, 0.5), (THREAT_GAIN, 0.3)], false, 0.0)),
        ("acetylcholine", h(0.30, 8.0, vec![(PLASTICITY, 0.6), (GAIN, 0.4)], false, 0.0)),
        ("melatonin", h(0.20, 50.0, vec![(GAIN, -0.5), (VIGOR, -0.5), (EXPLORATION, -0.3)], true, 0.0)),
        ("thyroid", h(0.50, 200.0, vec![(VIGOR, 0.5), (GAIN, 0.3)], false, 0.0)),
        ("testosterone", h(0.40, 120.0, vec![(INHIBITION, -0.4), (THREAT_GAIN, 0.3), (VIGOR, 0.3)], false, 0.0)),
        ("estrogen", h(0.40, 90.0, vec![(PLASTICITY, 0.4), (SOCIALITY, 0.2)], false, 0.0)),
        ("endorphin", h(0.20, 15.0, vec![(THREAT_GAIN, -0.4), (SOCIALITY, 0.3)], false, 0.0)),
    ]
}

pub struct Neuroendocrine {
    table: Vec<(&'static str, Hormone)>,
    pub level: Vec<f64>,
}

impl Neuroendocrine {
    pub fn new() -> Self {
        let table = default_table();
        let level = table.iter().map(|(_, h)| h.baseline).collect();
        Neuroendocrine { table, level }
    }
    pub fn with_table(table: Vec<(&'static str, Hormone)>) -> Self {
        let level = table.iter().map(|(_, h)| h.baseline).collect();
        Neuroendocrine { table, level }
    }
    fn idx(&self, name: &str) -> usize {
        self.table.iter().position(|(n, _)| *n == name).unwrap()
    }
    pub fn release(&mut self, name: &str, amount: f64) {
        let i = self.idx(name);
        self.level[i] += amount;
    }
    pub fn tick(&mut self, phase: f64) {
        for (i, (_, hor)) in self.table.iter().enumerate() {
            let mut target = hor.baseline;
            if hor.circadian {
                target = hor.baseline + 0.6 * 0.5 * (1.0 - (2.0 * PI * phase).cos());
            }
            self.level[i] += (target - self.level[i]) / hor.tau;
            if hor.feedback > 0.0 {
                self.level[i] -= hor.feedback * (self.level[i] - hor.baseline).max(0.0);
            }
        }
    }
    pub fn modulation(&self) -> [f64; N_KNOBS] {
        let mut knobs = [1.0; N_KNOBS];
        for (i, (_, hor)) in self.table.iter().enumerate() {
            let dev = self.level[i] - hor.baseline;
            for &(k, w) in &hor.couples {
                knobs[k] += w * dev;
            }
        }
        knobs.iter_mut().for_each(|v| *v = v.max(0.0));
        knobs
    }
}

impl Default for Neuroendocrine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hormone_orchestra_from_a_data_table() {
        // every hormone shifts its documented knobs (emergent from the coupling row)
        let sigs = [
            ("cortisol", THREAT_GAIN, 1.0), ("cortisol", TIME_HORIZON, -1.0), ("cortisol", PLASTICITY, -1.0),
            ("oxytocin", SOCIALITY, 1.0), ("oxytocin", THREAT_GAIN, -1.0),
            ("norepinephrine", GAIN, 1.0), ("norepinephrine", EXPLORATION, 1.0),
            ("serotonin", TIME_HORIZON, 1.0), ("serotonin", INHIBITION, 1.0),
            ("acetylcholine", PLASTICITY, 1.0), ("melatonin", GAIN, -1.0), ("thyroid", VIGOR, 1.0),
        ];
        for &(h, k, sign) in &sigs {
            let mut ne = Neuroendocrine::new();
            ne.release(h, 1.0);
            assert!(sign * (ne.modulation()[k] - 1.0) > 0.05, "{} should move knob {} sign {}", h, k, sign);
        }

        // opponency emerges: a knob is the SUM over hormones (oxytocin cancels cortisol's threat)
        let mut c = Neuroendocrine::new(); c.release("cortisol", 1.0);
        let mut co = Neuroendocrine::new(); co.release("cortisol", 1.0); co.release("oxytocin", 1.0);
        assert!(co.modulation()[THREAT_GAIN] < c.modulation()[THREAT_GAIN] - 0.3, "opponency");

        // different timescales from each tau: fast NE clears, slow thyroid lingers
        let mut ne3 = Neuroendocrine::new(); ne3.release("norepinephrine", 1.0); ne3.release("thyroid", 1.0);
        for _ in 0..25 {
            ne3.tick(0.0);
        }
        assert!(ne3.level[4] - 0.30 < 0.2 && ne3.level[7] - 0.50 > 0.8, "timescales"); // idx 4=NE, 7=thyroid
    }

    #[test]
    fn circadian_and_not_hardcoded() {
        // circadian gain: lower at night than by day, from the melatonin row
        let mut day = Neuroendocrine::new();
        let mut night = Neuroendocrine::new();
        for _ in 0..400 {
            day.tick(0.0);
            night.tick(0.5);
        }
        assert!(night.modulation()[GAIN] < day.modulation()[GAIN] - 0.1, "circadian");

        // NOT hardcoded (a): zero the couplings → the body is inert
        let inert_table: Vec<(&str, Hormone)> = default_table().into_iter()
            .map(|(n, mut h)| { h.couples = vec![]; (n, h) })
            .collect();
        let mut inert = Neuroendocrine::with_table(inert_table);
        for name in ["cortisol", "oxytocin", "norepinephrine"] {
            inert.release(name, 1.0);
        }
        assert!(inert.modulation().iter().all(|&v| (v - 1.0).abs() < 1e-9), "zero table → inert");

        // NOT hardcoded (b): flip ONE coupling's sign → cortisol now CALMS threat
        let mut flipped = default_table();
        let cort = &mut flipped[0].1;
        for c in cort.couples.iter_mut() {
            if c.0 == THREAT_GAIN {
                c.1 = -0.6;
            }
        }
        let mut rewired = Neuroendocrine::with_table(flipped);
        rewired.release("cortisol", 1.0);
        assert!(rewired.modulation()[THREAT_GAIN] < 1.0, "editing data reprograms the effect");
    }
}
