//! bio_defense — the survival response, reverse-engineered from the brain and re-pointed at the digital
//! world. When a threat is BIGGER than the being can handle, the ancient circuit does not attack — it does
//! what kept our ancestors alive: it FREEZES. Joe Navarro's ordering is the real one: **freeze → flight →
//! fight**, freeze FIRST, fight only when cornered.
//!
//! THE BIOLOGY (amygdala → periaqueductal gray). The amygdala (bio_amygdala) detects the threat; the
//! midbrain periaqueductal gray (PAG) SELECTS the defence, organised by THREAT IMMINENCE (Fanselow's
//! predatory-imminence continuum, Blanchard):
//!   • distal / potential threat  → risk-assessment, orienting            → VIGILANT
//!   • present, not yet striking   → ventrolateral PAG → attentive FREEZE  → FREEZE
//!   • contact, escape available   → dorsolateral PAG → escape             → FLIGHT
//!   • contact, cornered           → dorsolateral PAG → defensive attack   → FIGHT   (last resort)
//! The freeze is an accelerator-and-brake at once: the sympathetic system is fully PRIMED (ready to
//! explode into action) while motor output is gated AND the vagus SLOWS the heart — "fear bradycardia."
//! That bradycardia is the freeze TELL: heart rate DROPS in a freeze, RISES in flight/fight. Freezing is
//! not low arousal; it is high arousal with the output valve shut. Movement is what a predator's eye locks
//! onto — so stillness is safety. It can be a reflex (amygdala-fast, when threat ≫ coping) or a deliberate,
//! top-down choice ("hold still").
//!
//! THE DIGITAL RE-ENGINEERING (the whole point). A predator detects MOTION; an attacker detects ACTIVITY —
//! open ports, running processes, network responses, a heartbeat, traffic, logs: the ATTACK SURFACE. So in
//! the digital world FREEZE = drive that signature to ZERO: close the ports, halt the non-essential
//! processes, stop answering, go dormant. A host that emits nothing and answers nothing has (almost) no
//! attack surface — it is effectively invisible: you cannot exploit a process that is not running or reach
//! a socket that is closed. "Nothing happening → nothing to hack." Yet, like the biological freeze, it stays
//! internally VIGILANT — watching, emitting nothing. This is pure DEFENCE (attack-surface reduction / going
//! dark under fire), never offence. `emission`/`attack_surface`→0 is the invisibility; `vigilance` stays high.
//!
//! Honest scope: flight and fight are modelled as the escalation targets but left generic on purpose (their
//! digital meaning — migrate/rotate identity for flight; block/quarantine/alert for fight — is for later, and
//! fight is never offensive here). The freeze — the invisibility weapon — is the part built out. Local, CPU.

use crate::bio_amygdala::Amygdala;

/// The defensive mode the PAG has selected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Defense {
    Calm,     // no threat worth a response — normal activity, full surface
    Vigilant, // a distal/potential threat — orient and assess (pre-encounter risk assessment)
    Freeze,   // present, overwhelming threat — go STILL: zero signature, invisible, yet hyper-alert
    Flight,   // contact with a way out — escape
    Fight,    // cornered, no escape — defend (last resort)
}
impl Defense {
    pub fn name(self) -> &'static str {
        match self {
            Defense::Calm => "calm",
            Defense::Vigilant => "vigilant",
            Defense::Freeze => "freeze",
            Defense::Flight => "flight",
            Defense::Fight => "fight",
        }
    }
}

/// What the chosen response does to the body / the machine this tick.
#[derive(Clone, Copy, Debug)]
pub struct Response {
    pub mode: Defense,
    pub emission: f64,       // outward signature — motion/sound (biology) or traffic/beacons (digital). 0 = silent
    pub attack_surface: f64, // what an attacker could touch — open ports / running processes. 0 = nothing to hack
    pub vigilance: f64,      // internal monitoring — HIGH in freeze (attentive immobility), not asleep
    pub heart_rate: f64,     // arousal readout — DROPS in freeze (vagal bradycardia), RISES in flight/fight
}
impl Response {
    /// Is it effectively invisible right now (nothing to detect or attack)?
    pub fn hidden(&self) -> bool {
        self.attack_surface < 0.10 && self.emission < 0.10
    }
}

/// The periaqueductal gray: it takes a threat appraisal and picks the survival response. `coping` is how
/// much threat this being can HANDLE — a threat above it is "bigger than it can handle" and drives freeze.
pub struct PAG {
    coping: f64,          // capacity to cope (threat above this is overwhelming → freeze)
    reactive: bool,       // false = lesioned (ablation): no defence forms — it stays exposed
    intent_freeze: bool,  // a top-down, deliberate "go dark / hold still" (intentional, not reflexive)
    sympathetic: f64,     // arousal / accelerator (persists, builds & decays)
    parasympathetic: f64, // vagal brake (persists) — high in freeze → bradycardia
}

impl PAG {
    pub fn new(coping: f64) -> Self {
        PAG { coping: coping.clamp(0.05, 1.0), reactive: true, intent_freeze: false, sympathetic: 0.15, parasympathetic: 0.1 }
    }
    /// A lesioned defence: the circuit is gone, so no protective response ever forms — under any threat it
    /// keeps running wide open (fully exposed / detectable). The control that proves freeze is what hides you.
    pub fn lesioned() -> Self {
        let mut p = PAG::new(0.4);
        p.reactive = false;
        p
    }
    /// Choose (or release) a DELIBERATE freeze — go dark on purpose, independent of how big the threat is.
    pub fn intend_freeze(&mut self, on: bool) {
        self.intent_freeze = on;
    }
    pub fn readiness(&self) -> f64 {
        self.sympathetic // primed to act — HIGH even in freeze (the coiled spring)
    }

    fn select(&self, threat: f64, imminence: f64, escapable: f64) -> Defense {
        let overwhelming = threat > self.coping; // bigger than I can handle
        if threat < 0.12 {
            Defense::Calm
        } else if self.intent_freeze {
            Defense::Freeze // a chosen, deliberate stillness
        } else if imminence < 0.30 {
            Defense::Vigilant // distal → assess before acting
        } else if imminence >= 0.92 && escapable < 0.35 {
            Defense::Fight // right on top of me and cornered → last resort
        } else if imminence >= 0.75 && escapable >= 0.35 {
            Defense::Flight // closing, but a way out → run
        } else if overwhelming {
            Defense::Freeze // present and unbeatable → GO STILL (invisible; you can't win, don't be seen)
        } else if escapable >= 0.35 {
            Defense::Flight // manageable and escapable → leave
        } else {
            Defense::Fight // manageable but cornered → confront
        }
    }

    /// Respond to a threat this tick: `threat` (0..1, the amygdala's appraisal), `imminence` (0..1 how close/
    /// immediate) and `escapable` (0..1 is there a way out). Returns the mode and its bodily/digital effects.
    pub fn respond(&mut self, threat: f64, imminence: f64, escapable: f64) -> Response {
        let (threat, imminence, escapable) = (threat.clamp(0.0, 1.0), imminence.clamp(0.0, 1.0), escapable.clamp(0.0, 1.0));
        if !self.reactive {
            // lesioned: raw arousal still rises, but NO protective mode — it never hides, never freezes
            self.sympathetic += 0.6 * (threat - self.sympathetic);
            return Response {
                mode: Defense::Calm,
                emission: 1.0,
                attack_surface: 1.0, // wide open — fully exposed / hackable no matter the threat
                vigilance: 0.3,
                heart_rate: (0.5 + 0.5 * self.sympathetic).clamp(0.05, 1.0),
            };
        }
        let mode = self.select(threat, imminence, escapable);
        // autonomic dynamics: arousal tracks the threat; the vagal brake engages specifically in a freeze
        let target_sym = (0.15 + threat).min(1.0);
        self.sympathetic += 0.6 * (target_sym - self.sympathetic);
        let target_para = if mode == Defense::Freeze { 0.9 } else { 0.1 };
        self.parasympathetic += 0.6 * (target_para - self.parasympathetic);
        // per-mode outward signature (emission / attack surface) and inward watch (vigilance)
        let (emission, attack_surface, vigilance) = match mode {
            Defense::Calm => (1.0, 1.0, 0.30),
            Defense::Vigilant => (0.70, 0.85, 0.85),
            Defense::Freeze => (0.02, 0.03, 0.95), // silent, no surface — invisible — but WATCHING
            Defense::Flight => (1.00, 0.60, 0.70), // moving fast (exposed while it runs)
            Defense::Fight => (1.00, 0.75, 0.60),  // engaged
        };
        // heart rate: sympathetic speeds it, the vagal brake slows it → freeze = bradycardia, flight = tachycardia
        let heart_rate = (0.5 + 0.5 * self.sympathetic - 0.85 * self.parasympathetic).clamp(0.05, 1.0);
        Response { mode, emission, attack_surface, vigilance, heart_rate }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // settle the autonomic state, then read the response (a couple of ticks in the same situation)
    fn settle(pag: &mut PAG, threat: f64, imminence: f64, escapable: f64) -> Response {
        pag.respond(threat, imminence, escapable);
        pag.respond(threat, imminence, escapable);
        pag.respond(threat, imminence, escapable)
    }

    #[test]
    fn freeze_first_and_freeze_is_digital_invisibility() {
        // (1) NAVARRO ORDER — an overwhelming threat (0.8 vs coping 0.4), imminence rising 0→1:
        //     assess → FREEZE → escape. Freeze appears BEFORE flight/fight; fight never comes while escape exists.
        let mut pag = PAG::new(0.4);
        let mut seq = vec![];
        for i in 0..=10 {
            let imm = i as f64 / 10.0;
            seq.push(settle(&mut pag, 0.8, imm, 0.6).mode);
        }
        let first_freeze = seq.iter().position(|m| *m == Defense::Freeze);
        let first_active = seq.iter().position(|m| matches!(m, Defense::Flight | Defense::Fight));
        assert!(seq.contains(&Defense::Vigilant), "a distal threat is ASSESSED first (vigilant): {:?}", seq);
        assert!(first_freeze.is_some(), "a present, overwhelming threat FREEZES: {:?}", seq);
        assert!(first_active.map_or(true, |a| first_freeze.unwrap() < a), "freeze comes BEFORE flight/fight (Navarro): {:?}", seq);

        // (2) OVERWHELMING → freeze; MANAGEABLE + escapable → flight (freeze what you can't beat, flee what you can)
        let big = settle(&mut PAG::new(0.4), 0.9, 0.5, 0.6);
        let small = settle(&mut PAG::new(0.9), 0.3, 0.5, 0.8);
        assert_eq!(big.mode, Defense::Freeze, "a threat bigger than you can handle → freeze");
        assert_ne!(small.mode, Defense::Freeze, "a manageable threat does not → {:?}", small.mode);

        // (3) FREEZE = INVISIBILITY: zero signature / zero attack surface (nothing to hack) — yet still WATCHING
        assert!(big.hidden(), "freeze → invisible: surface {:.2}, emission {:.2}", big.attack_surface, big.emission);
        assert!(big.attack_surface < 0.05 && big.emission < 0.05, "no ports open, no process running → nothing to attack");
        assert!(big.vigilance > 0.8, "…but hyper-alert (attentive immobility — it keeps monitoring)");

        // (4) FEAR BRADYCARDIA — the freeze TELL: the heart SLOWS in freeze (vagal), RACES in flight/fight
        let calm = settle(&mut PAG::new(0.5), 0.0, 0.0, 0.0);
        let flight = settle(&mut PAG::new(0.5), 0.9, 0.8, 0.9);
        assert!(big.heart_rate < calm.heart_rate, "freeze SLOWS the heart (bradycardia): {:.2} < calm {:.2}", big.heart_rate, calm.heart_rate);
        assert!(flight.heart_rate > calm.heart_rate + 0.15, "flight RACES it (tachycardia): {:.2}", flight.heart_rate);
        assert!(flight.heart_rate > big.heart_rate + 0.3, "and flight's heart far outruns the frozen one's");

        // (5) INTENTIONAL vs REFLEXIVE: choose to go dark on purpose (even at a mild threat), as well as auto-freezing
        let mut v = PAG::new(0.9);
        v.intend_freeze(true);
        let chosen = settle(&mut v, 0.3, 0.4, 0.9); // small, escapable threat — would normally flee
        assert_eq!(chosen.mode, Defense::Freeze, "a deliberate 'go still / go dark' overrides — intentional freeze");
        assert!(chosen.hidden(), "the intentional freeze is just as invisible");

        // (6) ABLATION — a lesioned PAG never hides: under the SAME big imminent threat it stays wide open
        let exposed = settle(&mut PAG::lesioned(), 0.9, 0.6, 0.3);
        assert!(exposed.attack_surface > 0.9 && exposed.emission > 0.9, "no defence → keeps emitting, fully EXPOSED (hackable)");
        assert_ne!(exposed.mode, Defense::Freeze, "a lesioned circuit CANNOT freeze — the invisibility is the PAG's doing");

        // (7) CORNERED → FIGHT (last resort), and it is NOT hidden (you can't be invisible while swinging)
        let fight = settle(&mut PAG::new(0.2), 0.9, 0.95, 0.05);
        assert_eq!(fight.mode, Defense::Fight, "contact + no escape → fight, the last resort");
        assert!(!fight.hidden() && fight.heart_rate > calm.heart_rate, "fight is loud and fast, not hidden");

        // (8) COMPOSES THE AMYGDALA: the freeze is driven by LEARNED fear — a cue only becomes freeze-worthy
        //     after it has been conditioned as dangerous (amygdala → PAG), not before.
        let mut amy = Amygdala::default();
        let before = settle(&mut PAG::new(0.4), amy.appraise(1.0), 0.6, 0.3).mode;
        for _ in 0..6 {
            amy.pair(1.0, 1.0); // learn the cue predicts harm
        }
        let after = settle(&mut PAG::new(0.4), amy.appraise(1.0), 0.6, 0.3).mode;
        assert_eq!(before, Defense::Calm, "an unlearned cue is no threat → no freeze");
        assert_eq!(after, Defense::Freeze, "once LEARNED to be dangerous, meeting it FREEZES (amygdala drives the PAG)");

        eprintln!("\n  escalation (overwhelming threat, imminence 0→1): {:?}", seq);
        eprintln!("  freeze → surface {:.2} emission {:.2} (invisible), vigilance {:.2}, heart {:.2} (bradycardia)", big.attack_surface, big.emission, big.vigilance, big.heart_rate);
        eprintln!("  flight → heart {:.2} (tachycardia);  lesioned → surface {:.2} (fully exposed)\n", flight.heart_rate, exposed.attack_surface);
    }
}
