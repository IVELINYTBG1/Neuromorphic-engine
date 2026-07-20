//! bio_logic (curriculum #2 — LOGIC): deduction. Reasoning validly means drawing exactly what follows —
//! no less (completeness of the chain) and no MORE (soundness). Two operators, both already in the
//! substrate, are all it takes, and this cell is the engine that wields them:
//!   • IMPLICATION / modus ponens — from `a` and `a ⇒ b`, conclude `b`, and chain it (a ⇒ b ⇒ c …).
//!     That is bio_causal's transitive inference, now over truth rather than expectation.
//!   • CONJUNCTION — an antecedent like `cold AND wet ⇒ ice` must fire ONLY when BOTH hold. Getting this
//!     right is exactly the bio_net→bio_conj lesson: a rule that treats AND as "any of these" (OR)
//!     over-fires and reasons UNSOUNDLY (concludes ice from cold alone). Conjunction is load-bearing.
//!   • NEGATION — a literal may be negated (`cold AND NOT shelter ⇒ freeze`); closed-world (unknown =
//!     false), so adding a fact can RETRACT a conclusion (non-monotonic — the hallmark of real inference).
//! Forward-chaining computes the closure of everything entailed. This is the reasoning the being will run
//! over its grounded concepts, and the logic its node-designs will be checked against. Local, no backprop.

use std::collections::HashSet;

pub struct Literal {
    atom: String,
    positive: bool, // false = NOT atom
}

pub struct Rule {
    antecedent: Vec<Literal>, // a CONJUNCTION of literals
    consequent: String,
}

pub struct Logic {
    rules: Vec<Rule>,
    conjunctive: bool, // false = ablation: fire a rule when ANY antecedent literal holds (OR) → unsound
}

impl Logic {
    pub fn new(conjunctive: bool) -> Self {
        Logic { rules: vec![], conjunctive }
    }

    /// Add a rule: (atom, is_positive)…  ⇒  consequent.
    pub fn rule(&mut self, antecedent: &[(&str, bool)], consequent: &str) {
        self.rules.push(Rule {
            antecedent: antecedent.iter().map(|&(a, p)| Literal { atom: a.to_string(), positive: p }).collect(),
            consequent: consequent.to_string(),
        });
    }

    fn holds(&self, lit: &Literal, known: &HashSet<String>) -> bool {
        let present = known.contains(&lit.atom);
        if lit.positive { present } else { !present } // closed-world negation
    }

    fn fires(&self, r: &Rule, known: &HashSet<String>) -> bool {
        if self.conjunctive {
            r.antecedent.iter().all(|l| self.holds(l, known)) // AND — all must hold
        } else {
            r.antecedent.iter().any(|l| self.holds(l, known)) // ablation: OR — any is enough (unsound)
        }
    }

    /// Forward-chain from the given facts to the CLOSURE of everything entailed.
    pub fn deduce(&self, facts: &[&str]) -> HashSet<String> {
        let mut known: HashSet<String> = facts.iter().map(|s| s.to_string()).collect();
        loop {
            let mut added = false;
            for r in &self.rules {
                if !known.contains(&r.consequent) && self.fires(r, &known) {
                    known.insert(r.consequent.clone());
                    added = true;
                }
            }
            if !added {
                break;
            }
        }
        known
    }

    /// Does the knowledge base + these facts ENTAIL the query?
    pub fn entails(&self, facts: &[&str], q: &str) -> bool {
        self.deduce(facts).contains(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small knowledge base. `shelter` is a base fact (never a consequent), so negating it is stable.
    fn kb(conjunctive: bool) -> Logic {
        let mut l = Logic::new(conjunctive);
        l.rule(&[("rain", true)], "wet"); // rain ⇒ wet
        l.rule(&[("wet", true)], "cold"); // wet ⇒ cold
        l.rule(&[("cold", true), ("wet", true)], "ice"); // cold AND wet ⇒ ice
        l.rule(&[("cold", true), ("shelter", false)], "freeze"); // cold AND NOT shelter ⇒ freeze
        l
    }

    #[test]
    fn sound_deduction_needs_conjunction() {
        let l = kb(true);

        // (1) MODUS PONENS + CHAINING: rain ⊢ wet ⊢ cold (a two-step deduction never stated directly).
        assert!(l.entails(&["rain"], "cold"), "rain ⇒ wet ⇒ cold");
        assert!(l.entails(&["rain"], "ice"), "…and then cold AND wet ⇒ ice");

        // (2) CONJUNCTION keeps it SOUND: from cold ALONE, ice does NOT follow (wet is missing) — the AND
        //     prevents an invalid leap. With both, it does.
        assert!(!l.entails(&["cold"], "ice"), "cold alone must not entail ice (wet is absent)");
        assert!(l.entails(&["cold", "wet"], "ice"), "cold AND wet does entail ice");

        // (3) NEGATION & NON-MONOTONICITY: cold (no shelter) ⊢ freeze; add shelter and freeze is RETRACTED.
        assert!(l.entails(&["cold"], "freeze"), "cold AND NOT shelter ⇒ freeze");
        assert!(!l.entails(&["cold", "shelter"], "freeze"), "adding shelter retracts freeze (non-monotonic)");

        // (4) SOUNDNESS: it concludes nothing unentailed — no atom it was never given a route to.
        let closure = l.deduce(&["rain"]);
        assert!(!closure.contains("shelter") && !closure.contains("sun"), "derives only what follows: {:?}", closure);

        // (5) ABLATION — treat AND as OR and deduction becomes UNSOUND: ice is (wrongly) concluded from
        //     cold alone. Conjunction is what makes the reasoning valid.
        let sloppy = kb(false);
        assert!(sloppy.entails(&["cold"], "ice"), "OR-treatment over-fires: cold alone 'proves' ice (unsound)");

        let mut sorted: Vec<_> = l.deduce(&["rain"]).into_iter().collect();
        sorted.sort();
        eprintln!("\n  rain ⊢ {:?}   (chained: wet, cold, then ice & freeze)", sorted);
        eprintln!("  cold alone ⊬ ice (sound): {}", l.entails(&["cold"], "ice"));
        eprintln!("  cold ⊢ freeze, but cold+shelter ⊬ freeze (non-monotonic): {} / {}",
                  l.entails(&["cold"], "freeze"), l.entails(&["cold", "shelter"], "freeze"));
        eprintln!("  treat AND as OR → cold 'proves' ice (UNSOUND): {}\n", sloppy.entails(&["cold"], "ice"));
    }
}
