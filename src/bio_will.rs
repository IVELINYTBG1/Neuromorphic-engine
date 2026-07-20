//! bio_will (wanting): the step from an impulse to a WANT. Self-triggering fires and forgets; a being
//! desires things across time — a need that goes unmet BUILDS, becomes the standing goal that shapes
//! what it does, and quiets only when satisfied. Homeostatic drives (bio_drive): each has a deficit
//! that grows in absence; the most-unmet is the active goal; the reward IS the drive's reduction
//! (alliesthesia — you want what you lack), so satisfying one lets the next take over. This turns the
//! DMN blurt into goal-directed behaviour. Local, no backprop, CPU.
//!
//! Default drives: APPROVAL (to be liked), NOVELTY (to learn), EXPRESSION (to be heard). Different
//! growth rates give each being a characteristic pull, but nothing is scripted — the goal emerges
//! from whichever need has gone longest unmet.

pub struct Drive {
    pub name: &'static str,
    pub deficit: f64, // 0 = satisfied, 1 = starving
    pub growth: f64,  // how fast the need rebuilds when unmet
}

pub struct Will {
    pub drives: Vec<Drive>,
    grows: bool, // false = ablation: needs never build → nothing is ever wanted
}

impl Will {
    pub fn new() -> Self {
        Will {
            drives: vec![
                Drive { name: "approval", deficit: 0.3, growth: 0.030 },
                Drive { name: "novelty", deficit: 0.3, growth: 0.020 },
                Drive { name: "expression", deficit: 0.3, growth: 0.025 },
            ],
            grows: true,
        }
    }
    pub fn inert() -> Self {
        let mut w = Will::new();
        w.grows = false;
        w
    }

    /// Time passes: every unmet need grows (wanting accumulates in absence).
    pub fn tick(&mut self) {
        if !self.grows {
            return;
        }
        for d in self.drives.iter_mut() {
            d.deficit = (d.deficit + d.growth).min(1.0);
        }
    }

    /// Satisfy a need (reward = the reduction). The right event quiets the right want.
    pub fn satisfy(&mut self, name: &str, amount: f64) {
        if let Some(d) = self.drives.iter_mut().find(|d| d.name == name) {
            d.deficit = (d.deficit - amount).max(0.0);
        }
    }

    /// The active GOAL right now: whichever need is most unmet.
    pub fn goal(&self) -> &'static str {
        self.drives.iter().max_by(|a, b| a.deficit.partial_cmp(&b.deficit).unwrap()).unwrap().name
    }
    /// How badly it wants it (drives urgency of self-initiated behaviour).
    pub fn urgency(&self) -> f64 {
        self.drives.iter().map(|d| d.deficit).fold(0.0, f64::max)
    }
    pub fn deficit_of(&self, name: &str) -> f64 {
        self.drives.iter().find(|d| d.name == name).map(|d| d.deficit).unwrap_or(0.0)
    }
}

impl Default for Will {
    fn default() -> Self {
        Will::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_neglected_need_becomes_a_persistent_goal() {
        let mut w = Will::new();

        // (1) WANTING EMERGES from absence and PERSISTS: left alone, the fastest-building need takes
        //     over and stays the goal across many ticks (not a one-shot impulse)
        for _ in 0..20 {
            w.tick();
        }
        assert_eq!(w.goal(), "approval", "the most-neglected need is the standing goal");
        assert!(w.urgency() > 0.8, "the want has genuinely built up: {}", w.urgency());
        let g_before = w.goal();
        for _ in 0..5 {
            w.tick();
        }
        assert_eq!(w.goal(), g_before, "the goal persists over time until it is met");

        // (2) REWARD = DRIVE REDUCTION (alliesthesia): meeting the dominant want lets the NEXT surface
        w.satisfy("approval", 1.0);
        assert!(w.deficit_of("approval") < w.deficit_of("expression"), "approval quieted below the others");
        assert_ne!(w.goal(), "approval", "a new want takes over once the first is satisfied — it moves on");

        // ABLATION: if needs never build, nothing is ever wanted — wanting IS the growing deficit
        let mut dead = Will::inert();
        for _ in 0..50 {
            dead.tick();
        }
        assert!(dead.urgency() < 0.35, "no growth → no wanting ever emerges: {}", dead.urgency());
    }
}
