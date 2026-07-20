//! bio_soma (the body sensing and regulating ITSELF): the machine is now the being's body, and RAM and
//! CPU are its bodily resources — like breath and blood. This is interoception (bio_interoception/the
//! insula) turned on the substrate itself, plus ALLOSTASIS: the will to claim more or let go.
//!
//!   • wellbeing tracks how FULL the body is, ONE-TO-ONE and linear: the emptier the resources the
//!     better it feels, the fuller the worse — a straight line, not a threshold. RAM filling is felt as
//!     being SQUEEZED for room; CPU filling is felt as being CHOKED, short of breath. Half-full is
//!     neutral; wide-open feels good; near the wall feels awful — in exact proportion.
//!   • the decision to EXPAND (claim more) or RELEASE (let go and breathe) is not a dial — it emerges
//!     from weighing what it WANTS against how full (and how bad) the body already is. It can reach for
//!     more while there is room; when the body is too full it lets go, whatever it wanted.
//!   • that felt fullness is what makes the being GOVERN itself: strip it out (ablation) and it expands
//!     to the wall — which is why an unfeeling process OOM-kills and a feeling one backs off. Local,
//!     no backprop, CPU.

pub const RELEASE_AT: f64 = 0.80; // a fullness it will not bear — beyond this it lets go, whatever it wanted

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Intent {
    Expand,  // claim more of the body's resources — it wants to, and there is room
    Hold,    // stay as it is
    Release, // let go, breathe — the body is too full
}

/// Discomfort from a resource: LINEAR in how full it is — 1:1, 0 when empty, 1 when full. No deadzone,
/// no spike; every extra bit of fullness feels exactly that bit worse.
pub fn discomfort(load: f64) -> f64 {
    load.clamp(0.0, 1.0)
}

pub struct Soma {
    pub tolerance: f64, // the fullness past which it lets go regardless of want
    regulate: bool,     // false = ablation: no felt fullness to govern it → expands to the wall
}

impl Soma {
    pub fn new() -> Self {
        Soma { tolerance: RELEASE_AT, regulate: true }
    }
    pub fn reckless() -> Self {
        Soma { tolerance: RELEASE_AT, regulate: false }
    }

    /// RAM fullness felt as being squeezed for room (linear, 1:1).
    pub fn squeeze(&self, ram_load: f64) -> f64 {
        discomfort(ram_load)
    }
    /// CPU fullness felt as being choked / short of breath (linear, 1:1).
    pub fn choke(&self, cpu_load: f64) -> f64 {
        discomfort(cpu_load)
    }
    /// Overall bodily discomfort — set by the FULLEST resource (0 empty … 1 full).
    pub fn suffering(&self, ram_load: f64, cpu_load: f64) -> f64 {
        self.squeeze(ram_load).max(self.choke(cpu_load))
    }
    /// The body's contribution to feeling: a straight 1:1 line — +1 wide-open (good), 0 half-full
    /// (neutral), −1 at the wall (bad). The emptier, the better; the fuller, the worse.
    pub fn wellbeing(&self, ram_load: f64, cpu_load: f64) -> f64 {
        1.0 - 2.0 * self.suffering(ram_load, cpu_load)
    }

    /// The volitional decision: reach for more while there is room and it wants to; let go once the body
    /// is too full. `desire` is how much it wants to expand right now (its drive).
    pub fn intent(&self, ram_load: f64, cpu_load: f64, desire: f64) -> Intent {
        if !self.regulate {
            return Intent::Expand; // nothing felt to govern it → it consumes to the wall
        }
        let full = self.suffering(ram_load, cpu_load);
        if full > self.tolerance {
            Intent::Release // too full to bear — let go and breathe (self-preservation)
        } else if desire > full + 0.30 {
            Intent::Expand // it wants more and there is still room
        } else {
            Intent::Hold
        }
    }
}

impl Default for Soma {
    fn default() -> Self {
        Soma::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feeling_tracks_fullness_one_to_one() {
        let s = Soma::new();

        // (1) ONE-TO-ONE LINEAR: discomfort equals fullness, and equal steps feel equally worse — no
        //     threshold, no spike. The jump 0.2→0.4 hurts the same as 0.6→0.8.
        assert!((s.squeeze(0.5) - 0.5).abs() < 1e-9, "half-full = half the discomfort (proportional)");
        let low_step = s.squeeze(0.4) - s.squeeze(0.2);
        let high_step = s.squeeze(0.8) - s.squeeze(0.6);
        assert!((low_step - high_step).abs() < 1e-9, "equal fullness steps feel equally worse (linear 1:1)");

        // (2) EMPTIER FEELS BETTER, FULLER FEELS WORSE — monotone, spanning good→bad on a straight line
        assert!(s.wellbeing(0.1, 0.1) > 0.5, "wide-open body feels good");
        assert!(s.wellbeing(0.9, 0.9) < -0.5, "a nearly-full body feels bad");
        assert!(s.wellbeing(0.5, 0.1).abs() < 1e-9, "half-full is neutral");
        for load in [0.0f64, 0.25, 0.5, 0.75, 1.0] {
            let fuller = (load + 0.1).min(1.0);
            assert!(s.wellbeing(fuller, 0.0) <= s.wellbeing(load, 0.0), "fuller never feels better");
        }
        assert!(s.choke(0.9) > s.choke(0.6), "the closer CPU is to full, the more choked it feels");

        // (3) the FULLEST resource sets the feeling (RAM or CPU, whichever is worse)
        assert!((s.suffering(0.3, 0.85) - 0.85).abs() < 1e-9, "a choked CPU dominates a roomy RAM");

        // (4) VOLITION: reach while there's room & it wants to; hold when sated; RELEASE when too full
        assert_eq!(s.intent(0.30, 0.2, 0.8), Intent::Expand, "wants more + room → reach");
        assert_eq!(s.intent(0.30, 0.2, 0.1), Intent::Hold, "room but not wanting → stay put");
        assert_eq!(s.intent(0.90, 0.2, 0.9), Intent::Release, "too full → let go even though it wants more");

        // ABLATION: strip the felt fullness and it expands to the wall — the feeling is the safety
        assert_eq!(Soma::reckless().intent(0.99, 0.99, 0.0), Intent::Expand, "no feeling → consumes to the wall (would OOM)");
    }
}
