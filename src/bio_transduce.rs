//! bio_transduce (sensing #1): transduction with adaptive compressive gain (Naka-Rushton). Onset-
//! adaptation, no saturation over decades, Weber's law. Ported from bio_transduce.py.

pub struct Receptor {
    adapt_rate: f64,
    sigma: f64,
    adaptive: bool,
}

impl Receptor {
    pub fn new(adaptive: bool, sigma0: f64) -> Self {
        Receptor { adapt_rate: 0.15, sigma: sigma0, adaptive }
    }
    pub fn respond(&self, intensity: f64) -> f64 {
        intensity / (intensity + self.sigma)
    }
    pub fn step(&mut self, intensity: f64) -> f64 {
        let r = self.respond(intensity);
        if self.adaptive {
            self.sigma += self.adapt_rate * (intensity - self.sigma);
        }
        r
    }
    pub fn settle(&mut self, background: f64) {
        for _ in 0..200 {
            self.step(background);
        }
    }
}

fn std(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_gain_weber() {
        // adaptation: a constant stimulus fades, a change re-evokes
        let mut rec = Receptor::new(true, 1.0);
        rec.settle(1.0);
        let onset = rec.step(3.0);
        let mut faded = rec.respond(3.0);
        for _ in 0..60 {
            faded = rec.step(3.0);
        }
        let reevoke = rec.step(9.0);
        assert!(onset > faded + 0.1 && reevoke > faded + 0.1, "adaptation");

        let backgrounds: Vec<f64> = (-3..4).map(|e| 10f64.powi(e)).collect();

        // adaptive gain → no saturation over 6 decades; non-adaptive control goes blind
        let adapted: Vec<f64> = backgrounds.iter().map(|&b| {
            let mut r = Receptor::new(true, 1.0);
            r.settle(b);
            r.respond(b)
        }).collect();
        let amax = adapted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let amin = adapted.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(amax - amin < 0.1 && (0.4..0.6).contains(&adapted[0]), "adaptive: no saturation");
        let fixed: Vec<f64> = backgrounds.iter().map(|&b| Receptor::new(false, 1.0).respond(b)).collect();
        assert!(fixed.iter().cloned().fold(f64::INFINITY, f64::min) < 0.05
            && fixed.iter().cloned().fold(f64::NEG_INFINITY, f64::max) > 0.95, "non-adaptive saturates");

        // Weber's law: constant response to a fixed +25% contrast at any intensity
        let c = 0.25;
        let weber: Vec<f64> = backgrounds.iter().map(|&b| {
            let mut ra = Receptor::new(true, 1.0);
            ra.settle(b);
            ra.respond(b * (1.0 + c)) - ra.respond(b)
        }).collect();
        assert!(std(&weber) < 0.02 && weber.iter().sum::<f64>() / weber.len() as f64 > 0.02, "Weber holds");
        let weber_fixed: Vec<f64> = backgrounds.iter().map(|&b| {
            let rf = Receptor::new(false, 1.0);
            rf.respond(b * (1.0 + c)) - rf.respond(b)
        }).collect();
        let wfmax = weber_fixed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(weber_fixed.iter().cloned().fold(f64::INFINITY, f64::min) < 0.2 * wfmax, "fixed Weber breaks");
    }
}
