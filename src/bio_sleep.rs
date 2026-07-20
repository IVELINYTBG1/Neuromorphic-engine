//! bio_sleep (memory #7): complementary learning systems — fast hippocampus + slow cortex + sleep replay.
//! Cortex learns from hippocampal replay; lesion the hippocampus → consolidated survive, recent lost.
//! Ported from bio_sleep.py.

use crate::rng::Rng;
use crate::vec;

pub struct AssocStore {
    d: usize,
    w: Vec<f64>, // d×d
}

impl AssocStore {
    pub fn new(d: usize) -> Self {
        AssocStore { d, w: vec![0.0; d * d] }
    }
    pub fn learn(&mut self, cue: &[f64], target: &[f64], lr: f64) {
        vec::add_outer(&mut self.w, target, cue, lr);
    }
    pub fn recall(&self, cue: &[f64]) -> Vec<f64> {
        vec::mv(&self.w, cue, self.d, self.d)
            .iter()
            .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
            .collect()
    }
    pub fn lesion(&mut self) {
        self.w.iter_mut().for_each(|w| *w = 0.0);
    }
}

fn overlap(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systems_consolidation_by_replay() {
        let (d, k) = (120, 6);
        let mut g = Rng::new(0);
        let bits = |g: &mut Rng| -> Vec<f64> { (0..d).map(|_| (g.randint(2) as f64) * 2.0 - 1.0).collect() };
        let cues: Vec<Vec<f64>> = (0..k).map(|_| bits(&mut g)).collect();
        let targets: Vec<Vec<f64>> = (0..k).map(|_| bits(&mut g)).collect();

        let mut hippo = AssocStore::new(d);
        let mut cortex = AssocStore::new(d);
        for i in 0..k {
            hippo.learn(&cues[i], &targets[i], 1.0);
        }
        let hippo_recall = (0..k).map(|i| overlap(&hippo.recall(&cues[i]), &targets[i])).sum::<f64>() / k as f64;
        let cortex_before = (0..k).map(|i| overlap(&cortex.recall(&cues[i]), &targets[i])).sum::<f64>() / k as f64;
        assert!(hippo_recall > 0.95, "hippocampus one-shot");
        assert!(cortex_before < 0.75, "cortex naive");

        // sleep replay: hippocampus reinstates episodes, cortex slowly learns them
        for _ in 0..25 {
            for i in 0..k {
                let replay = hippo.recall(&cues[i]);
                cortex.learn(&cues[i], &replay, 0.08);
            }
        }
        let cortex_after = (0..k).map(|i| overlap(&cortex.recall(&cues[i]), &targets[i])).sum::<f64>() / k as f64;
        assert!(cortex_after > 0.95, "consolidated by replay");

        // a recent memory learned after the last sleep, then lesion the hippocampus
        let recent_cue = bits(&mut g);
        let recent_target = bits(&mut g);
        hippo.learn(&recent_cue, &recent_target, 1.0);
        hippo.lesion();
        let consolidated = (0..k).map(|i| overlap(&cortex.recall(&cues[i]), &targets[i])).sum::<f64>() / k as f64;
        let recent = overlap(&cortex.recall(&recent_cue), &recent_target);
        assert!(consolidated > 0.95 && recent < 0.75, "consolidated survive, recent lost (amnesia)");
    }
}
