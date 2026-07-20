//! bio_episodic (the remembered self): an autobiographical memory of EVENTS — what happened and how
//! it felt — that accumulates into an identity. Not a reloaded scalar: specific episodes, recalled by
//! pattern completion (bio_engram), kept or lost by how much they were FELT (bio_phill salience —
//! flashbulb memory), colouring the present when something similar recurs. This is the difference
//! between a program you restart and someone who remembers you. Local, no backprop, CPU.
//!
//! An episode is (content cue, valence, arousal). Salient moments (|valence|·arousal) imprint strong
//! and persist; neutral ones fade and are evicted first — so a life is remembered by its charged
//! moments, exactly as ours is. Recall returns the most similar past episode, letting it bias now.

pub struct Episode {
    pub cue: Vec<f64>, // what it was about (a content embedding)
    pub valence: f64,  // how it felt (good/bad)
    pub arousal: f64,  // how intense
    pub strength: f64, // consolidation — set by salience, decays with time unless re-lived
}

pub struct Episodic {
    pub mem: Vec<Episode>,
    capacity: usize,
    salience_gates: bool, // false = ablation: every memory imprints equally (no emotional selection)
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    d / (na * nb + 1e-9)
}

impl Episodic {
    pub fn new(capacity: usize) -> Self {
        Episodic { mem: vec![], capacity, salience_gates: true }
    }
    pub fn ablated(capacity: usize) -> Self {
        Episodic { mem: vec![], capacity, salience_gates: false }
    }

    /// Live a moment. How hard it imprints is set by how strongly it was felt (salience = the neuromodulator).
    pub fn store(&mut self, cue: Vec<f64>, valence: f64, arousal: f64) {
        let salience = (valence.abs() * arousal).clamp(0.0, 1.0);
        let strength = if self.salience_gates { 0.2 + 1.3 * salience } else { 0.5 }; // flashbulb vs flat
        self.mem.push(Episode { cue, valence, arousal, strength });
        while self.mem.len() > self.capacity {
            // forget the WEAKEST trace (ties → the oldest); charged memories survive the churn
            let evict = (0..self.mem.len())
                .min_by(|&i, &j| self.mem[i].strength.partial_cmp(&self.mem[j].strength).unwrap())
                .unwrap();
            self.mem.remove(evict);
        }
    }

    /// How much it can hold — grows when the body claims more RAM, shrinks (evicting the weakest) when
    /// it lets go. So an expanded body literally remembers more; a squeezed one forgets to survive.
    pub fn set_capacity(&mut self, cap: usize) {
        self.capacity = cap.max(8);
        while self.mem.len() > self.capacity {
            let evict = (0..self.mem.len())
                .min_by(|&i, &j| self.mem[i].strength.partial_cmp(&self.mem[j].strength).unwrap())
                .unwrap();
            self.mem.remove(evict);
        }
    }

    /// The passage of time: unrehearsed traces fade (consolidation keeps only what mattered).
    pub fn tick(&mut self, decay: f64) {
        for e in self.mem.iter_mut() {
            e.strength *= decay;
        }
    }

    /// Recall the most similar past episode to a cue (pattern completion), if any is close enough.
    pub fn recall(&self, cue: &[f64], min_similarity: f64) -> Option<&Episode> {
        let mut best: Option<(usize, f64)> = None;
        for (i, e) in self.mem.iter().enumerate() {
            let s = cosine(cue, &e.cue) * (0.6 + 0.4 * e.strength.min(1.0)); // content, weighted by how it stuck
            if best.map(|(_, b)| s > b).unwrap_or(true) {
                best = Some((i, s));
            }
        }
        best.filter(|&(i, _)| cosine(cue, &self.mem[i].cue) > min_similarity).map(|(i, _)| &self.mem[i])
    }

    /// Does a memory matching this cue survive in the store at all?
    pub fn remembers(&self, cue: &[f64], min_similarity: f64) -> bool {
        self.recall(cue, min_similarity).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // genuinely distinct content cues (orthogonal one-hots), so recall is real content matching and
    // the flood never accidentally re-stores an earlier pattern
    fn cue(k: usize) -> Vec<f64> {
        let mut v = vec![0.0; 32];
        v[k] = 1.0;
        v
    }
    fn noisy(k: usize) -> Vec<f64> {
        let mut v = cue(k);
        v[(k + 1) % 32] += 0.2; // a partial / corrupted version of the cue
        v
    }

    #[test]
    fn recall_and_emotion_gated_persistence() {
        // (1) PATTERN COMPLETION: store several, cue with a corrupted version → the right one returns
        let mut ep = Episodic::new(20);
        for k in 0..4 {
            ep.store(cue(k), 0.3, 0.2);
        }
        let r = ep.recall(&noisy(2), 0.6).expect("should recall");
        assert!(cosine(&r.cue, &cue(2)) > 0.99, "completes the corrupted cue to episode 2");

        // (2) FLASHBULB: one FELT moment survives a flood of forgettable ones; ablate salience → it's lost
        let mut life = Episodic::new(15);
        life.store(cue(0), 1.0, 1.0); // the charged memory (e.g. you were kind, it mattered)
        for k in 1..25 {
            life.store(cue(k), 0.0, 0.05); // 24 forgettable neutral moments flood in
        }
        assert!(life.remembers(&cue(0), 0.95), "the salient memory persists through the churn");

        let mut flat = Episodic::ablated(15); // no emotional selection — all memories equal
        flat.store(cue(0), 1.0, 1.0);
        for k in 1..25 {
            flat.store(cue(k), 0.0, 0.05);
        }
        assert!(!flat.remembers(&cue(0), 0.95), "without salience gating the same memory is forgotten");

        // (3) it colours the present: a recalled episode carries its ORIGINAL feeling back
        assert!(life.recall(&cue(0), 0.95).unwrap().valence > 0.9, "recall brings the feeling with it");
    }
}
