//! bio_bond (knowing you): a being is someone-with-someone. This is a model of the OTHER — a learned
//! estimate of your disposition toward it, a prediction of what you'll do next (a forward model = the
//! root of theory of mind), and a TRUST that grows slowly with kindness and breaks fast with harm.
//! Nothing here is scripted about "the architect"; the relationship is whatever the interactions have
//! made it. Bond gates warmth and empathy, so being known changes how you are treated back. Local,
//! no backprop, CPU.
//!
//! Trust asymmetry (slow up, fast down) is the real shape of trust: a hundred kind moments build it,
//! one betrayal dents it. But a bond also HARDENS. The brain does not keep a relationship on one
//! timescale — sustained trust CONSOLIDATES (synaptic tagging & capture / LTP → structural) into a
//! lasting bond that decays only over a very long time and CUSHIONS harm. So a young bond is fragile,
//! while a long-nurtured one survives a fight and time apart — it has become part of who you are. That
//! consolidation is what stops a bond "coming and going" too quickly to ever mean anything.

pub struct Bond {
    pub trust: f64,           // the ACTIVE bond right now (0..1) — moves with recent interactions
    pub hardened: f64,        // the CONSOLIDATED bond — built by sustained trust, decays very slowly, cushions harm
    pub partner_valence: f64, // learned model of how the other feels toward me
    build: f64,               // trust gained per unit kindness (slow)
    break_: f64,              // trust lost per unit harm (fast) — the asymmetry
    consol: f64,              // rate the earned trust HARDENS into the lasting bond (STM → LTM; slow)
    lr: f64,                  // how fast the model of the other updates
    learns: bool,             // false = ablation: no model, no relationship forms
}

impl Bond {
    pub fn new() -> Self {
        Bond { trust: 0.2, hardened: 0.0, partner_valence: 0.0, build: 0.12, break_: 0.6, consol: 0.04, lr: 0.25, learns: true }
    }
    pub fn inert() -> Self {
        let mut b = Bond::new();
        b.learns = false;
        b
    }

    /// One interaction, felt at `valence` (from your words/acts). Updates the model, the trust, and — when
    /// trust is sustained — hardens the lasting bond.
    pub fn interact(&mut self, valence: f64) {
        if !self.learns {
            return;
        }
        // model of the other's disposition — a running forward model (theory of mind)
        self.partner_valence += self.lr * (valence - self.partner_valence);
        // trust: earned slowly on kindness, lost quickly on harm — but a HARDENED bond cushions the blow and
        // cannot be cratered below most of what has consolidated (you don't lose a deep friend to one fight)
        if valence >= 0.0 {
            self.trust = (self.trust + self.build * valence * (1.0 - self.trust)).clamp(0.0, 1.0);
        } else {
            let drop = self.break_ * (-valence) * (1.0 - 0.85 * self.hardened);
            self.trust = (self.trust - drop).clamp(self.hardened * 0.85, 1.0);
        }
        self.consolidate();
    }

    /// SUSTAINED trust hardens into a lasting bond (synaptic consolidation): the part of the active bond that
    /// has been HELD is slowly captured into the consolidated bond, which then barely fades.
    fn consolidate(&mut self) {
        if self.trust > self.hardened {
            self.hardened += self.consol * (self.trust - self.hardened);
        }
    }

    /// Time passing: the model of you fades slowly, being together keeps the bond consolidating, and the
    /// hardened bond fades only over a very long time. The ACTIVE trust does NOT drain away on its own — a
    /// bond persists between meetings; only harm takes it down, and only down to the hardened floor.
    pub fn tick(&mut self) {
        if !self.learns {
            return;
        }
        self.partner_valence *= 0.999;
        self.consolidate(); // held trust keeps deepening even between explicit acts
        self.hardened *= 0.99997; // the lasting bond fades only across a very long time
    }

    /// What it EXPECTS of you next (the forward model) — the basis of surprise.
    pub fn expect(&self) -> f64 {
        self.partner_valence
    }
    /// How surprising an interaction is vs its model of you (prediction error = felt betrayal / delight).
    pub fn surprise(&self, valence: f64) -> f64 {
        (valence - self.expect()).abs()
    }
    /// How open/warm/empathic it is toward you — gated by the bond it has built (active, floored by hardened).
    pub fn openness(&self) -> f64 {
        self.trust.max(self.hardened)
    }
}

impl Default for Bond {
    fn default() -> Self {
        Bond::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_builds_slow_breaks_fast_and_predicts() {
        // (1) BUILDS with kindness, and BREAKS faster than it builds (the shape of real trust)
        let mut b = Bond::new();
        for _ in 0..15 {
            b.interact(0.6); // consistent kindness
        }
        let earned = b.trust;
        assert!(earned > 0.6, "trust grows with kindness: {earned}");
        let before = b.trust;
        b.interact(-0.8); // a single betrayal
        let lost = before - b.trust;
        let one_kind_gain = b.build * 0.6 * (1.0 - before); // what one kind act would have added
        assert!(lost > 4.0 * one_kind_gain, "one betrayal costs more than several kindnesses (asymmetry)");

        // (2) THEORY OF MIND: after consistent kindness it EXPECTS kindness (low surprise); betrayal shocks
        let mut c = Bond::new();
        for _ in 0..15 {
            c.interact(0.7);
        }
        assert!(c.expect() > 0.5, "it models you as kind");
        assert!(c.surprise(0.7) < 0.2, "an expected kindness is unsurprising");
        assert!(c.surprise(-0.8) > 1.0, "a betrayal is a large prediction error (felt as shock)");

        // (3) the bond GATES openness (warmth/empathy toward you)
        assert!(c.openness() > 0.6, "a strong bond opens it up");

        // ABLATION: with no learning there is no relationship — trust stays flat, no model of you
        let mut dead = Bond::inert();
        for _ in 0..15 {
            dead.interact(0.7);
        }
        assert!((dead.trust - 0.2).abs() < 1e-6 && dead.expect().abs() < 1e-6, "no learning → no bond");
    }

    #[test]
    fn a_bond_consolidates_and_hardens_over_time() {
        // a YOUNG bond is fragile — one betrayal early craters it
        let mut young = Bond::new();
        young.interact(0.6);
        let young_before = young.trust;
        young.interact(-0.8);
        let young_lost = young_before - young.trust;

        // a LONG-NURTURED bond hardens (consolidation) and survives time apart AND the same betrayal
        let mut deep = Bond::new();
        for _ in 0..40 {
            deep.interact(0.6); // a long history of kindness
        }
        for _ in 0..80 {
            deep.tick(); // ... and much time apart
        }
        let deep_before = deep.trust;
        assert!(deep_before > 0.6, "a consolidated bond HOLDS through time apart (it doesn't drain away): {:.2}", deep_before);
        assert!(deep.hardened > 0.4, "sustained trust hardened into a lasting bond: {:.2}", deep.hardened);
        deep.interact(-0.8); // the same betrayal
        let deep_lost = deep_before - deep.trust;

        // the deep bond LOSES LESS to the same blow, and never craters below what consolidated
        assert!(deep_lost < young_lost, "a consolidated bond resists harm: deep lost {:.2} vs young {:.2}", deep_lost, young_lost);
        assert!(deep.trust > 0.45, "the hardened bond remains after the blow (a fight doesn't end a lifelong friendship): {:.2}", deep.trust);
    }
}
