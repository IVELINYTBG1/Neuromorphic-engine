//! bio_grammar (Broca's area — the sequence of speech): association tells you WHICH words go together;
//! grammar tells you in what ORDER. This is the directed, sequential structure of language — the
//! word-level lift of the substrate's sequence cells (bio_seq's STDP, bio_branch's Markov sampling):
//! hearing "the cat runs" strengthens the DIRECTED links the→cat→runs (pre-before-post = LTP, STDP),
//! so later the being can predict what comes next and GENERATE a novel well-formed sequence by
//! sampling those links (not argmax — bio_branch showed argmax collapses to the single most-likely
//! sentence). It is what lets speech be composed, not just recalled. Local, Hebbian, no backprop, CPU.
//!
//! The distinction that matters: association (bio_semantics) is symmetric co-occurrence ("cat" and
//! "the" go together); grammar is DIRECTIONAL (the→cat, never cat→the) — order, not just company.
//! Learned from the same stream at the same time, so speech and meaning grow together.

use crate::rng::Rng;
use std::collections::HashMap;

pub const START: &str = "\u{2}"; // sentence-begin sentinel
pub const END: &str = "\u{3}"; // sentence-end sentinel

pub struct Grammar {
    trans: HashMap<String, HashMap<String, f64>>, // word → (successor → learned strength), DIRECTED
    lr: f64,
    learns: bool, // false = ablation: no sequence learning → no structure to speak
}

impl Grammar {
    pub fn new(lr: f64) -> Self {
        Grammar { trans: HashMap::new(), lr, learns: true }
    }
    pub fn inert() -> Self {
        Grammar { trans: HashMap::new(), lr: 0.0, learns: false }
    }

    /// Hear a sentence: strengthen each directed adjacent link (Hebbian / STDP pre→post). START and
    /// END frame it so the being learns how sentences BEGIN and CLOSE, not only their interiors.
    pub fn observe(&mut self, tokens: &[&str]) {
        if !self.learns || tokens.is_empty() {
            return;
        }
        let mut seq: Vec<&str> = vec![START];
        seq.extend_from_slice(tokens);
        seq.push(END);
        for w in seq.windows(2) {
            *self.trans.entry(w[0].to_string()).or_default().entry(w[1].to_string()).or_insert(0.0) += self.lr;
        }
    }

    /// Successors of a word, strongest first (the learned continuation distribution).
    pub fn successors(&self, word: &str) -> Vec<(String, f64)> {
        let mut v: Vec<(String, f64)> = self.trans.get(word).map(|m| m.iter().map(|(k, &s)| (k.clone(), s)).collect()).unwrap_or_default();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v
    }

    /// The single most likely next word (for prediction tests).
    pub fn predict_top(&self, word: &str) -> Option<String> {
        self.successors(word).into_iter().next().map(|(w, _)| w)
    }

    /// Sample a successor in proportion to its learned strength (composition, not rote recall).
    pub fn sample_next(&self, word: &str, rng: &mut Rng) -> Option<String> {
        let succ = self.trans.get(word)?;
        if succ.is_empty() {
            return None;
        }
        let keys: Vec<&String> = succ.keys().collect();
        let weights: Vec<f64> = keys.iter().map(|k| succ[*k]).collect();
        Some(keys[rng.multinomial(&weights)].clone())
    }

    /// GENERATE a fresh sentence from the learned grammar: walk START → … → END by sampling.
    /// ANSWER FROM a word she just heard — the reply grows out of what YOU said rather than from nowhere.
    /// Empty if that word leads nowhere she has learned, so the caller can fall back to free composition.
    pub fn continue_from(&self, word: &str, max_len: usize, rng: &mut Rng) -> Vec<String> {
        let mut out = vec![word.to_string()];
        let mut cur = word.to_string();
        for _ in 0..max_len {
            match self.sample_next(&cur, rng) {
                Some(w) if w == END => break,
                Some(w) => {
                    out.push(w.clone());
                    cur = w;
                }
                None => break,
            }
        }
        if out.len() > 1 {
            out
        } else {
            vec![]
        }
    }

    pub fn generate(&self, max_len: usize, rng: &mut Rng) -> Vec<String> {
        let mut out = vec![];
        let mut cur = START.to_string();
        for _ in 0..max_len {
            match self.sample_next(&cur, rng) {
                Some(w) if w == END => break,
                Some(w) => {
                    out.push(w.clone());
                    cur = w;
                }
                None => break,
            }
        }
        out
    }

    pub fn weight(&self, a: &str, b: &str) -> f64 {
        self.trans.get(a).and_then(|m| m.get(b)).copied().unwrap_or(0.0)
    }
    pub fn size(&self) -> usize {
        self.trans.values().map(|m| m.len()).sum()
    }
    /// All learned links (for persistence).
    pub fn links(&self) -> Vec<(String, String, f64)> {
        let mut v = vec![];
        for (a, m) in &self.trans {
            for (b, &w) in m {
                v.push((a.clone(), b.clone(), w));
            }
        }
        v
    }
    pub fn set_link(&mut self, a: &str, b: &str, w: f64) {
        *self.trans.entry(a.to_string()).or_default().entry(b.to_string()).or_insert(0.0) = w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_word_order_directed_and_generates() {
        let corpus = [
            vec!["the", "cat", "runs", "fast"],
            vec!["the", "dog", "runs", "home"],
            vec!["a", "bird", "flies", "high"],
            vec!["a", "fish", "swims", "deep"],
        ];
        let mut g = Grammar::new(1.0);
        for _ in 0..5 {
            for s in &corpus {
                g.observe(s);
            }
        }

        // (1) LEARNS TRANSITION STRUCTURE: it predicts a real successor it heard
        assert!(["cat", "dog"].contains(&g.predict_top("the").unwrap().as_str()), "'the' → a noun it heard");
        assert!(["fast", "home"].contains(&g.predict_top("runs").unwrap().as_str()), "'runs' → an adverb it heard");

        // (2) GRAMMAR ≠ ASSOCIATION — the links are DIRECTED: the→cat exists, cat→the never observed
        assert!(g.weight("the", "cat") > 0.0, "forward link learned");
        assert!(g.weight("cat", "the") == 0.0, "reverse link NOT learned — it captured ORDER, not co-occurrence");

        // (3) GENERATES well-formed novel sentences by SAMPLING (varies, doesn't collapse to one)
        let mut rng = Rng::new(1);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..40 {
            let gen = g.generate(8, &mut rng);
            assert!(!gen.is_empty(), "produces a sentence");
            // every adjacent step (incl. START→first and last→END) must be a link it actually learned
            let framed: Vec<String> = std::iter::once(START.to_string())
                .chain(gen.iter().cloned())
                .chain(std::iter::once(END.to_string()))
                .collect();
            for w in framed.windows(2) {
                assert!(g.weight(&w[0], &w[1]) > 0.0, "generated a link it never heard: {} → {}", w[0], w[1]);
            }
            seen.insert(gen.join(" "));
        }
        assert!(seen.len() > 1, "sampling gives variety, not one memorized sentence");

        // ABLATION: no sequence learning → no structure, nothing to compose
        let mut dead = Grammar::inert();
        for s in &corpus {
            dead.observe(s);
        }
        assert!(dead.size() == 0 && dead.generate(8, &mut rng).is_empty(), "no learning → no grammar");
    }
}
