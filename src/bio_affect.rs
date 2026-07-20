//! bio_affect (limbic #3): core affect = valence×arousal — a fast emotion riding a slow mood that
//! integrates it and bends the next appraisal (mood-congruent). Ported from bio_affect.py.

use crate::vec::clamp;

pub struct CoreAffect {
    pub val_e: f64,
    pub aro_e: f64,
    pub val_m: f64,
    pub aro_m: f64,
    emo_decay: f64,
    mood_track: f64,
    congruence: f64,
}

impl CoreAffect {
    pub fn new(congruence: f64) -> Self {
        CoreAffect { val_e: 0.0, aro_e: 0.0, val_m: 0.0, aro_m: 0.0,
                     emo_decay: 0.5, mood_track: 0.08, congruence }
    }
    pub fn appraise(&self, valence: f64) -> f64 {
        clamp(valence + self.congruence * self.val_m, -1.0, 1.0)
    }
    pub fn event(&mut self, valence: f64, arousal: f64) {
        self.val_e = clamp(self.val_e + self.appraise(valence), -1.0, 1.0);
        self.aro_e = clamp(self.aro_e + arousal, -1.0, 1.0);
    }
    pub fn tick(&mut self) {
        self.val_m = clamp(self.val_m + self.mood_track * (self.val_e - self.val_m), -1.0, 1.0);
        self.aro_m = clamp(self.aro_m + self.mood_track * (self.aro_e - self.aro_m), -1.0, 1.0);
        self.val_e *= self.emo_decay;
        self.aro_e *= self.emo_decay;
    }
}

impl Default for CoreAffect {
    fn default() -> Self {
        CoreAffect::new(0.6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valence_arousal_plane_and_mood() {
        // events map onto the valence×arousal plane
        let mut a = CoreAffect::default(); a.event(1.0, 0.2);
        let (rv, ra) = (a.val_e, a.aro_e);
        let mut a = CoreAffect::default(); a.event(-1.0, 1.0);
        let (tv, ta) = (a.val_e, a.aro_e);
        assert!(rv > 0.5 && tv < -0.5 && ta > ra + 0.5);

        // fast emotion vs slow lingering mood (two clocks)
        let mut a = CoreAffect::default(); a.event(1.0, 0.0);
        for _ in 0..6 { a.tick(); }
        let (emo_after, mood_after) = (a.val_e, a.val_m);
        assert!(emo_after < 0.1);
        let mut b = CoreAffect::default();
        for _ in 0..15 { b.event(0.6, 0.0); b.tick(); }
        let mood_sustained = b.val_m;
        for _ in 0..8 { b.tick(); }
        assert!(mood_sustained > 3.0 * mood_after && b.val_e < 0.1 && b.val_m > 0.3,
                "mood lingers after emotion fades");

        // mood-congruent appraisal
        let read_neutral = CoreAffect::default().appraise(0.3);
        let mut grim = CoreAffect::default();
        for _ in 0..15 { grim.event(-0.8, 0.5); grim.tick(); }
        assert!(grim.appraise(0.3) < read_neutral - 0.1, "grim mood reads the same event more negatively");
    }
}
