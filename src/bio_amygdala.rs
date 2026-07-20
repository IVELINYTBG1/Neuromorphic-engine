//! bio_amygdala (limbic #2): fast fear conditioning + extinction as opponent SAFETY learning (not
//! erasure) → spontaneous recovery. Ported from bio_amygdala.py.

pub struct Amygdala {
    pub w_fear: f64,
    pub w_safe: f64,
    lr_acq: f64,
    lr_ext: f64,
}

impl Default for Amygdala {
    fn default() -> Self {
        Amygdala { w_fear: 0.0, w_safe: 0.0, lr_acq: 0.4, lr_ext: 0.3 }
    }
}

impl Amygdala {
    pub fn appraise(&self, cs: f64) -> f64 {
        (cs * (self.w_fear - self.w_safe)).max(0.0)
    }
    pub fn pair(&mut self, cs: f64, us: f64) {
        self.w_fear += self.lr_acq * cs * (us - self.w_fear);
    }
    pub fn extinguish(&mut self, cs: f64) {
        self.w_safe += self.lr_ext * cs * (self.w_fear - self.w_safe);
    }
    pub fn time_passes(&mut self, decay: f64) {
        self.w_safe *= decay;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fear_conditions_and_recovers() {
        let mut amy = Amygdala::default();
        assert!(amy.appraise(1.0) < 0.05, "neutral cue starts non-threatening");
        for _ in 0..5 {
            amy.pair(1.0, 1.0);
        }
        let acquired = amy.appraise(1.0);
        let fear_after_acq = amy.w_fear;
        assert!(acquired > 0.5, "fast fear acquisition");
        for _ in 0..20 {
            amy.extinguish(1.0);
        }
        assert!(amy.appraise(1.0) < 0.1, "extinction suppresses fear");
        assert!((amy.w_fear - fear_after_acq).abs() < 0.05, "fear trace not erased");
        amy.time_passes(0.4);
        assert!(amy.appraise(1.0) > 0.3, "spontaneous recovery");
    }
}
