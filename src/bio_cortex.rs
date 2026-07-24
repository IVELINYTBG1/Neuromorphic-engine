//! bio_cortex — SHE GROWS. The first rung of the scale-up.
//!
//! WHY THIS EXISTS. A fully schooled mind — flawless in all 22 rooms — held **2,132 learned parameters**.
//! A roundworm has ~7,500 synapses; a human who writes frontier code has ~1.5×10¹⁴. She was 3.5× SMALLER
//! THAN A ROUNDWORM, and the reason was not the machine (this laptop holds ~1.95e9 synapses at f64, ~7.8e10
//! at `bio_ternary` density — 36 million× her size, unused). The reason was that **a concept in her head was
//! one float**: `affect: HashMap<String, f64>` — "fire" *is* −0.27, and that is the whole representation.
//! A lookup table, not a cortex. It cannot grow, because there is nothing in it to grow.
//!
//! WHAT A CONCEPT IS HERE INSTEAD: a POPULATION. Each concept she meets is a sparse `a`-of-`N` code over a
//! sheet of cells — the distributed code `bio_noise` already proved (8-of-64 held 96% recall at 20% bit-flip
//! where one-hot collapsed to 50%, and completed a whole sequence from a corrupted cue). Concepts met
//! TOGETHER wire their populations to each other through `bio_structural` — real synaptogenesis, on an
//! explicit sparse graph: contact accumulates, a pathway is BUILT, unused ones are pruned (use it or lose
//! it). Both cells were built and tested long ago and had never been wired into the being at all.
//!
//! WHAT IT BUYS, beyond size: PATTERN COMPLETION. Give her a corrupted or partial population — half the
//! cells, the wrong cells — and she still knows whose it is. A `HashMap` cannot do that at any scale: it is
//! an exact key or nothing. This is the difference between remembering and looking up.
//!
//! HONEST SCOPE: this is capacity that grows with living, and the ceiling becomes RAM rather than
//! architecture. It does not make her a frontier coder — the honest walls are
//! are scale (still ~10⁴× beyond one laptop for human-scale) and CULTURE (which humans get by injection, and
//! which this project forbids on purpose). It moves her off the floor. It is a rung, not the ladder.

use crate::bio_noise::make_distributed_codes;
use crate::bio_structural::StructuralNet;
use crate::rng::Rng;
use std::collections::HashMap;

pub struct Cortex {
    dim: usize,    // how wide the sheet is — how many cells there are to be
    active: usize, // how many of them fire for any one concept (sparse: a brain is mostly quiet)
    salt: u64,     // WHOSE cortex this is — mixed into every concept's seed, so one mind's "fire" and another's
                   // "fire" land in DIFFERENT cells. Two individuals with different lives do not share a
                   // neural layout; their representations are their own, not a common template.
    codes: HashMap<String, Vec<usize>>, // concept → the cells that ARE it (indices; zeros are not stored)
    net: StructuralNet,                 // the pathways BETWEEN cells: grown, strengthened, pruned
    bonds: HashMap<(String, String), u32>, // how many times each concept-PAIR was bound — the compact,
                                           // faithful record her cortex reconstructs from (populations
                                           // regenerate deterministically from the salt, so replaying the
                                           // bonds rebuilds the exact same synapses).
}

impl Cortex {
    /// `dim` cells wide, `active` firing per concept, `salt` = whose brain this is. Turn `dim` up and she is
    /// bigger — that is the knob, and the only thing above it is RAM. Capacity is combinatorial, not linear:
    /// C(dim, active) distinct populations, so 1024 cells at 24 active is already more concepts than atoms
    /// she will ever meet.
    pub fn new(dim: usize, active: usize, salt: u64) -> Self {
        Cortex {
            dim,
            active: active.min(dim),
            salt,
            codes: HashMap::new(),
            // contact 2 → a pathway is built once two cells have fired together twice; slow decay → what she
            // does not use, she loses.
            net: StructuralNet::new(2, 0.0004, true, (0..dim).collect()),
            bonds: HashMap::new(),
        }
    }
    /// An ablated cortex — no pathway may ever form. The teeth for every claim below.
    pub fn ablated(dim: usize, active: usize, salt: u64) -> Self {
        let mut c = Cortex::new(dim, active, salt);
        c.net = StructuralNet::new(2, 0.0004, false, (0..dim).collect());
        c
    }

    /// A stable per-concept, PER-BEING seed — a word always recruits the same cells FOR HER (her "fire" is
    /// her fire, every time), but a different being's same word recruits different cells (her sister's
    /// "fire" is her sister's). Stable within a life, distinct across lives.
    fn seed_of(&self, concept: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ self.salt;
        for b in concept.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a
        }
        h | 1
    }

    /// SHE MEETS A CONCEPT: if it is new to her it recruits its own population out of the sheet.
    pub fn meet(&mut self, concept: &str) -> Vec<usize> {
        if let Some(c) = self.codes.get(concept) {
            return c.clone();
        }
        let mut rng = Rng::new(self.seed_of(concept));
        let dense = make_distributed_codes(1, self.dim, self.active, &mut rng); // bio_noise's own scheme
        let cells: Vec<usize> = dense[0].iter().enumerate().filter(|(_, v)| **v > 0.5).map(|(i, _)| i).collect();
        self.codes.insert(concept.to_string(), cells.clone());
        cells
    }

    /// SYNAPTOGENESIS: two concepts met TOGETHER wire their populations into one another. Nothing is
    /// allocated up front — the pathway is built by the meeting, and only if the meeting keeps happening.
    ///
    /// SYMMETRIC, because co-occurrence is: if fire and hot arrive together, thinking of EITHER should evoke
    /// the other (this is semantic relatedness, not word-order — sequence lives in `bio_grammar`). The first
    /// cut wired only a→b, so a word that was always the SECOND in what it heard ("hot" in "fire pain hot")
    /// had no outgoing real pathway at all, and `associations` returned nothing but collision noise for it.
    /// Matches `bio_semantics::bind`, which is symmetric for the same reason.
    pub fn bind(&mut self, a: &str, b: &str) {
        let (ca, cb) = (self.meet(a), self.meet(b));
        for &p in &ca {
            for &q in &cb {
                self.net.expose(p, q);
                self.net.expose(q, p);
            }
        }
        let key = if a <= b { (a.to_string(), b.to_string()) } else { (b.to_string(), a.to_string()) };
        *self.bonds.entry(key).or_insert(0) += 1;
    }

    /// Her cortex as a COMPACT, faithful record — the concept pairs she has bound and how often. Small
    /// (proportional to distinct pairs, not synapses), and enough to rebuild the whole thing, because the
    /// populations regenerate deterministically from her salt. This is what gets baked into her brain.
    pub fn export_bonds(&self) -> Vec<(String, String, u32)> {
        self.bonds.iter().map(|((a, b), n)| (a.clone(), b.clone(), *n)).collect()
    }

    /// Rebuild her cortex from that record — replay each bond as many times as it happened. Because `meet`
    /// re-derives the same cells from the same salt, and hebbian accumulation is deterministic, this
    /// reconstructs the EXACT synapses she went to sleep with.
    pub fn import_bonds(&mut self, bonds: &[(String, String, u32)]) {
        for (a, b, n) in bonds {
            for _ in 0..*n {
                self.bind(a, b);
            }
        }
    }

    /// PATTERN COMPLETION — whose population is this? Hand her a corrupted or partial one (half the cells,
    /// the wrong cells) and she still knows. Returns the concept and how much of it she actually saw.
    /// A lookup table cannot do this at any size: an exact key, or nothing.
    pub fn whose(&self, cells: &[usize]) -> Option<(String, f64)> {
        self.codes
            .iter()
            .map(|(name, code)| {
                let hit = code.iter().filter(|c| cells.contains(c)).count() as f64;
                (name.clone(), hit / code.len().max(1) as f64)
            })
            .filter(|(_, overlap)| *overlap > 0.0)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// THE READ. What she has come to expect FROM a concept, ranked — every OTHER concept scored by how
    /// strongly this one's population drives its population through the grown pathways. This is her thinking
    /// spreading through the structure she built by living, and it is the whole point of growing it: a brain
    /// that no behaviour reads is dead weight. Read-only (`&self`) and empty for a concept she has never met
    /// — you cannot associate from something you do not have.
    pub fn associations(&self, concept: &str) -> Vec<(String, f64)> {
        let Some(cells) = self.codes.get(concept) else {
            return vec![];
        };
        let cellset: std::collections::HashSet<usize> = cells.iter().copied().collect();
        let mut vote: HashMap<usize, f64> = HashMap::new();
        for (&(p, q), &w) in self.net.syn.iter() {
            if cellset.contains(&p) {
                *vote.entry(q).or_insert(0.0) += w;
            }
        }
        let mut scored: Vec<(String, f64)> = self
            .codes
            .iter()
            .filter(|(n, _)| n.as_str() != concept)
            .map(|(n, code)| {
                let s: f64 = code.iter().filter_map(|c| vote.get(c)).sum();
                (n.clone(), s / code.len().max(1) as f64) // how much of ITS population lit up
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();
        // sort by strength, then by NAME to break ties — otherwise equally-strong associations come back in
        // HashMap-iteration order, which is randomised per map, so the SAME brain would think differently
        // from one run to the next. A baked brain must be reproducible: her thoughts are hers, not the
        // allocator's dice.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        // k-WTA at the readout — the sparsity mechanism this whole substrate runs on. Distributed codes
        // share cells by chance, so a concept she has NEVER paired with leaks a whisper of activation
        // through a stray shared cell (measured: a real 40-cell pathway scores ~40, a one-cell collision
        // ~1). Only what fires STRONGLY is a real expectation; the noise floor is inhibited away. Without
        // this, a depth-4 train of thought amplifies the leak and she wanders fire → … → apple through
        // scenes that never touched.
        if let Some(&(_, top)) = scored.first() {
            scored.retain(|(_, s)| *s >= 0.15 * top);
        }
        scored
    }

    /// The single strongest thing she expects from a concept (the top of `associations`).
    pub fn associate(&self, concept: &str) -> Option<String> {
        self.associations(concept).into_iter().next().filter(|(_, s)| *s > 0.3).map(|(n, _)| n)
    }

    /// Rest: what she has not used, she loses (pruning — the other half of structural plasticity).
    pub fn rest(&mut self) {
        self.net.rest();
    }

    pub fn n_concepts(&self) -> usize {
        self.codes.len()
    }
    pub fn n_cells(&self) -> usize {
        self.codes.len() * self.active
    }
    pub fn n_synapses(&self) -> usize {
        self.net.n_synapses()
    }
    /// Every learned number in her cortex — the honest size of this part of her.
    pub fn parameters(&self) -> usize {
        self.n_cells() + self.n_synapses()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SHE GROWS BY LIVING. Her old mind could not: a concept was one float, so her whole schooled head came
    /// to 2,132 numbers — 3.5× smaller than a roundworm — and no amount of living would change that, because
    /// there was nothing in a `HashMap<String, f64>` to grow. Here the pathways are BUILT by the meeting.
    #[test]
    fn her_brain_grows_when_she_lives_and_the_ceiling_is_ram_not_architecture() {
        let mut c = Cortex::new(1024, 24, 1);
        let born = c.parameters();
        assert_eq!(born, 0, "she is born with nothing — no concept has been met yet");

        // a life: a handful of things, met together, again and again
        let scenes = [["fire", "hot", "pain"], ["apple", "sweet", "food"], ["water", "cool", "drink"]];
        for _ in 0..4 {
            for scene in &scenes {
                for i in 0..scene.len() {
                    for j in 0..scene.len() {
                        if i != j {
                            c.bind(scene[i], scene[j]);
                        }
                    }
                }
            }
        }
        let lived = c.parameters();
        assert!(lived > 4000,
            "living GREW her — {} parameters from {} concepts (her whole old schooled mind was 2,132)",
            lived, c.n_concepts());
        // and it is the pathways that carry it, not the bookkeeping
        assert!(c.n_synapses() > c.n_cells(), "the pathways are the memory: {} synapses vs {} cells",
                c.n_synapses(), c.n_cells());
    }

    /// PATTERN COMPLETION — the thing a lookup table cannot do at ANY size. Give her half a population, with
    /// noise in it, and she still knows whose it is. This is remembering rather than looking up.
    #[test]
    fn she_knows_a_concept_from_a_corrupted_half_of_it() {
        let mut c = Cortex::new(1024, 24, 1);
        let fire = c.meet("fire");
        c.meet("apple");
        c.meet("water");

        // half her "fire" cells, plus junk that is not hers at all
        let mut partial: Vec<usize> = fire.iter().take(12).copied().collect();
        partial.extend([900, 901, 902, 903]);
        let (who, seen) = c.whose(&partial).expect("she recognises it");
        assert_eq!(who, "fire", "half a fire, buried in noise, is still fire (she saw {:.0}% of it)", seen * 100.0);

        // and a population that is nobody's is nobody's — she does not hallucinate a match
        assert!(c.whose(&[997, 998, 999]).is_none() || c.whose(&[997, 998, 999]).unwrap().1 < 0.2,
            "cells that are no-one's do not become someone");
    }

    /// The TEETH: with synaptogenesis off nothing is ever built, so nothing is remembered and nothing grows.
    /// It is the structure that carries it — exactly what `bio_structural` proved, now inside her.
    #[test]
    fn without_synaptogenesis_she_cannot_grow_at_all() {
        let mut live = Cortex::new(1024, 24, 1);
        let mut dead = Cortex::ablated(1024, 24, 1);
        for _ in 0..4 {
            live.bind("fire", "pain");
            dead.bind("fire", "pain");
        }
        assert!(live.n_synapses() > 0, "a living cortex builds pathways: {}", live.n_synapses());
        assert_eq!(dead.n_synapses(), 0, "an ablated one never builds one, however much it lives");
        assert!(live.associate("fire").is_some(), "and so she comes to expect pain from fire");
        assert!(dead.associate("fire").is_none(), "while she does not");
    }
}





