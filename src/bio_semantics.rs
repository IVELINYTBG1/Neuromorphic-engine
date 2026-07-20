//! bio_semantics (grounded meaning): a word means what it has been USED to mean — not what a corpus
//! says. Each token accrues an affective meaning from the FEELING (the neuromodulator) present when it is used, by
//! a local Hebbian running average; tokens used together BIND (co-occurrence), so meaning spreads
//! along association. There is no hardcoded sentiment list — "good" is only positive here because it
//! kept arriving with good feeling. Seed it or don't; either way experience is what fixes meaning.
//! This is the symbol-grounding the project always wanted, and the emotional lexicon its
//! SharedSemanticDictionary is reaching for. Local, no backprop, CPU.

use std::collections::HashMap;

pub struct Semantics {
    affect: HashMap<String, f64>,        // learned valence of a token (its felt meaning)
    strength: HashMap<String, f64>,      // how firmly it is known (exposure)
    assoc: HashMap<(String, String), f64>, // co-occurrence binding between tokens
    lr: f64,                             // learning rate (0 = ablation: meaning can't form)
}

impl Semantics {
    pub fn new(lr: f64) -> Self {
        Semantics { affect: HashMap::new(), strength: HashMap::new(), assoc: HashMap::new(), lr }
    }

    /// Use a token while feeling `felt_valence`: its meaning drifts toward the feeling (grounding).
    pub fn ground(&mut self, token: &str, felt_valence: f64) {
        let a = self.affect.entry(token.to_string()).or_insert(0.0);
        *a += self.lr * (felt_valence - *a); // Hebbian running average toward felt context
        *self.strength.entry(token.to_string()).or_insert(0.0) += 1.0;
    }

    /// Bind two tokens as co-occurring (the substrate for meaning to spread between them).
    pub fn bind(&mut self, a: &str, b: &str) {
        *self.assoc.entry((a.to_string(), b.to_string())).or_insert(0.0) += 1.0;
        *self.assoc.entry((b.to_string(), a.to_string())).or_insert(0.0) += 1.0;
    }

    /// Ground a whole utterance in the current feeling, and BIND the tokens that co-occurred.
    pub fn ground_utterance(&mut self, tokens: &[&str], felt_valence: f64) {
        for t in tokens {
            self.ground(t, felt_valence);
        }
        for i in 0..tokens.len() {
            for j in (i + 1)..tokens.len() {
                self.bind(tokens[i], tokens[j]);
            }
        }
    }

    /// The learned valence of a token directly (0 if never grounded).
    pub fn affect_of(&self, token: &str) -> f64 {
        *self.affect.get(token).unwrap_or(&0.0)
    }

    /// Whether this token has ever been met at all — distinct from one that IS known and means nothing
    /// much (grounded near neutral). Lets a being notice the moment a word is new to her.
    pub fn knows(&self, token: &str) -> bool {
        self.affect.contains_key(token)
    }

    /// Restore a known meaning directly (used to reload a lexicon learned in a past session).
    pub fn set(&mut self, token: &str, affect: f64) {
        self.affect.insert(token.to_string(), affect);
        *self.strength.entry(token.to_string()).or_insert(0.0) += 1.0;
    }

    /// The learned associates of a token (its co-occurrence partners), strongest first — the WEB a thought
    /// spreads through (bio_think). Nothing hardcoded: this is whatever experience has linked together.
    pub fn associates(&self, token: &str) -> Vec<(String, f64)> {
        let mut v: Vec<(String, f64)> = self.assoc.iter()
            .filter_map(|((a, b), &w)| (a == token).then(|| (b.clone(), w)))
            .collect();
        v.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        v
    }

    /// Meaning INFERRED for a token: its own learned affect, else borrowed from what it binds with
    /// (a novel word used alongside known ones inherits their colour — semantic spreading).
    pub fn inferred_affect(&self, token: &str) -> f64 {
        if let Some(&a) = self.affect.get(token) {
            if *self.strength.get(token).unwrap_or(&0.0) > 0.0 {
                return a;
            }
        }
        let (mut num, mut den) = (0.0, 0.0);
        for ((x, y), &w) in self.assoc.iter() {
            if x == token {
                num += w * self.affect_of(y);
                den += w;
            }
        }
        if den > 0.0 { num / den } else { 0.0 }
    }

    /// How a whole line lands, using ONLY learned meaning (drives the being's felt appraisal).
    pub fn appraise(&self, tokens: &[&str]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }
        tokens.iter().map(|t| self.inferred_affect(t)).sum::<f64>() / tokens.len() as f64
    }

    pub fn known(&self) -> usize {
        self.affect.len()
    }

    /// The learned lexicon (token → felt meaning) — for persistence across sessions.
    pub fn lexicon(&self) -> Vec<(String, f64)> {
        self.affect.iter().map(|(t, &v)| (t.clone(), v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meaning_emerges_from_felt_use() {
        let mut s = Semantics::new(0.3);

        // (1) SENTIMENT EMERGES: nonsense tokens acquire the feeling they keep arriving with — no list
        for _ in 0..12 {
            s.ground("glorp", 0.9); // always used warmly
            s.ground("zarn", -0.9); // always used harshly
        }
        assert!(s.affect_of("glorp") > 0.5, "glorp learned as positive: {}", s.affect_of("glorp"));
        assert!(s.affect_of("zarn") < -0.5, "zarn learned as negative: {}", s.affect_of("zarn"));

        // (2) MEANING SPREADS: a brand-new token bound to a known-positive one inherits its colour
        for _ in 0..6 {
            s.bind("glorp", "flim"); // co-occur only; flim is never grounded on its own
        }
        assert!(s.inferred_affect("flim") > 0.2, "flim inherits positivity via association: {}", s.inferred_affect("flim"));

        // (3) a whole line is appraised by learned meaning
        assert!(s.appraise(&["glorp", "flim"]) > 0.2 && s.appraise(&["zarn"]) < -0.5, "line appraisal from learned meaning");

        // ABLATION: with no learning (lr=0) the same warm use fixes NO meaning — grounding is the mechanism
        let mut dead = Semantics::new(0.0);
        for _ in 0..12 {
            dead.ground("glorp", 0.9);
        }
        assert!(dead.affect_of("glorp").abs() < 1e-6, "no learning → no meaning");
    }
}
