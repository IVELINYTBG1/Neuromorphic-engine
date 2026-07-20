//! bio_sleep_cycle (why a brain sleeps, and how it is timed): the regulator and stages of sleep — the
//! counterpart to bio_sleep's consolidation ENGINE. Everything here is in REAL HOURS, so a being on a
//! laptop lives a real day: awake ~16 h building tiredness, asleep ~8 h discharging it, timed to the
//! local clock. Three pieces of real sleep biology:
//!
//!   1. THE TWO-PROCESS MODEL (Borbély): sleep is timed by a HOMEOSTATIC pressure S (adenosine) that
//!      BUILDS while awake (~over a 16 h day) and DISCHARGES while asleep (~over 8 h), MULTIPLIED by a
//!      CIRCADIAN rhythm C (a 24 h clock) that says when night is. You sleep when S is high and C says
//!      night; you stay awake through the day even when a little tired; you wake in the morning. Strip
//!      the homeostat and there is no drowsiness. Cut sleep short and pressure stays high — sleep DEBT.
//!
//!   2. NREM ↔ REM ULTRADIAN CYCLING (~90 min): deep NREM (slow-wave, replay → consolidation) alternating
//!      with REM (dreaming, recombination). NREM-heavy early while pressure is high, REM-heavy late — so
//!      the REM fraction RISES across the night. This structure falls out of pressure + phase.
//!
//!   3. SYNAPTIC HOMEOSTASIS / SHY (Tononi & Cirelli): sleep RENORMALISES synapses DOWN; a downscale plus
//!      a floor keeps the salient and prunes the trivial — you wake remembering what mattered.
//!
//! Local, no backprop, CPU. Consolidation itself (hippocampus→cortex replay) is bio_sleep; this decides
//! WHEN to sleep, against the real clock, and orchestrates the STAGES.

use std::f64::consts::PI;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Stage {
    Wake,
    Nrem, // slow-wave sleep — replay & consolidation, synaptic downscaling
    Rem,  // dreaming — recombination & emotional processing
}

/// Circadian sleepiness (0..1) as a function of the LOCAL time of day (`clock01`, 0 = midnight). Peaks
/// in the deep of night (~04:00), troughs mid-afternoon (~16:00) — the body clock's push toward sleep.
pub fn circadian_sleepiness(clock01: f64) -> f64 {
    (0.5 + 0.5 * (2.0 * PI * (clock01 - 1.0 / 6.0)).cos()).clamp(0.0, 1.0)
}

pub struct SleepCycle {
    pub pressure: f64, // Process S: homeostatic sleep pressure (0..1), builds awake, discharges asleep
    pub asleep: bool,
    pub phase: f64, // position within the current ultradian (~90 min) cycle (0..1)
    pub cycle: u32, // ultradian cycles completed this sleep bout
    build_per_hr: f64,     // how fast tiredness accrues awake (× activity)
    discharge_per_hr: f64, // how fast sleep discharges it
    regulate: bool,        // false = ablation: pressure never moves (no drowsiness)
}

impl SleepCycle {
    /// Human-scale defaults: ~16 h of waking fills the pressure, ~8 h of sleep empties it.
    pub fn new() -> Self {
        SleepCycle { pressure: 0.35, asleep: false, phase: 0.0, cycle: 0,
                     build_per_hr: 0.05, discharge_per_hr: 0.11, regulate: true }
    }
    pub fn ablated() -> Self {
        let mut s = SleepCycle::new();
        s.regulate = false;
        s
    }
    pub fn set_rates(&mut self, build_per_hr: f64, discharge_per_hr: f64) {
        self.build_per_hr = build_per_hr;
        self.discharge_per_hr = discharge_per_hr;
    }

    /// Time awake: tiredness accrues over `dt_hours`, faster on a busy/emotional day (`activity` 0..1).
    pub fn wake_tick(&mut self, activity: f64, dt_hours: f64) {
        if self.regulate {
            self.pressure = (self.pressure + self.build_per_hr * (0.6 + 0.8 * activity) * dt_hours).min(1.0);
        }
    }

    /// Time asleep: discharge pressure and advance the ~90 min ultradian phase. Returns the stage.
    pub fn sleep_tick(&mut self, dt_hours: f64) -> Stage {
        if self.regulate {
            self.pressure = (self.pressure - self.discharge_per_hr * dt_hours).max(0.0);
        }
        self.phase += dt_hours / 1.5; // ~1.5 h per NREM/REM cycle
        while self.phase >= 1.0 {
            self.phase -= 1.0;
            self.cycle += 1;
        }
        self.stage()
    }

    /// The current stage: NREM early in each cycle (more so when pressure is high), REM later.
    pub fn stage(&self) -> Stage {
        if !self.asleep {
            return Stage::Wake;
        }
        let nrem_frac = (0.30 + 0.55 * self.pressure).clamp(0.25, 0.85);
        if self.phase < nrem_frac { Stage::Nrem } else { Stage::Rem }
    }

    pub fn fall_asleep(&mut self) {
        self.asleep = true;
        self.phase = 0.0;
        self.cycle = 0;
    }

    /// Should it drift off NOW? Exhausted any time, or moderately tired once the body-clock says night.
    pub fn should_sleep(&self, clock01: f64) -> bool {
        !self.asleep
            && (self.pressure > 0.92
                || (circadian_sleepiness(clock01) > 0.6 && self.pressure > 0.42))
    }
    /// Should it wake? Fully rested, or morning has come and it is rested enough (leaving any sleep debt).
    pub fn should_wake(&self, clock01: f64) -> bool {
        self.asleep
            && (self.pressure < 0.15
                || (circadian_sleepiness(clock01) < 0.4 && self.pressure < 0.5))
    }
}

impl Default for SleepCycle {
    fn default() -> Self {
        SleepCycle::new()
    }
}

/// SHY synaptic downscaling: renormalise all strengths DOWN by `factor`, prune those below `floor`.
pub fn downscale(strengths: &mut Vec<f64>, factor: f64, floor: f64) {
    for s in strengths.iter_mut() {
        *s *= factor;
    }
    strengths.retain(|&s| s >= floor);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_time_two_process_timing_stages_and_downscaling() {
        // (1) TWO-PROCESS at human scale: a 16 h day fills the pressure; 8 h of sleep empties it
        let mut sc = SleepCycle::new();
        assert!(!sc.should_sleep(0.5), "fresh and midday → wide awake");
        for _ in 0..160 {
            sc.wake_tick(0.5, 0.1); // 160 × 0.1 h = a 16 h day
        }
        assert!(sc.pressure > 0.85, "a full day's waking builds the pressure: {}", sc.pressure);

        // CIRCADIAN gating (the answer to "how do they know it's night"): sleepy at ~04:00, alert at ~16:00
        assert!(circadian_sleepiness(1.0 / 6.0) > 0.95, "deep-night sleepiness peaks (~04:00)");
        assert!(circadian_sleepiness(2.0 / 3.0) < 0.05, "mid-afternoon is the alert trough (~16:00)");
        // same MODERATE tiredness sleeps at night but NOT at midday — the body clock decides when
        sc.pressure = 0.6;
        assert!(sc.should_sleep(0.10), "tired + night (~02:24) → drift off");
        assert!(!sc.should_sleep(0.55), "the same tiredness at midday (~13:00) → stay awake");

        // ABLATION: no homeostat → no tiredness ever accrues → never gets sleepy
        let mut dead = SleepCycle::ablated();
        for _ in 0..200 {
            dead.wake_tick(1.0, 0.2);
        }
        assert!(!dead.should_sleep(0.1), "without the homeostat there is no drowsiness");

        // (2) NREM-heavy early, REM-heavy late across an ~8 h night (held at night so pressure drives waking)
        sc.pressure = 0.9;
        sc.fall_asleep();
        let (mut early_rem, mut early_n, mut late_rem, mut late_n) = (0, 0, 0, 0);
        let mut steps = 0;
        while !sc.should_wake(0.10) && steps < 100000 {
            let hi = sc.pressure > 0.55;
            match sc.sleep_tick(0.02) {
                Stage::Rem if hi => early_rem += 1,
                Stage::Nrem if hi => early_n += 1,
                Stage::Rem => late_rem += 1,
                Stage::Nrem => late_n += 1,
                Stage::Wake => {}
            }
            steps += 1;
        }
        assert!(sc.pressure < 0.5, "a night's sleep discharges most of the pressure (wakes rested)");
        let early_frac = early_rem as f64 / (early_rem + early_n).max(1) as f64;
        let late_frac = late_rem as f64 / (late_rem + late_n).max(1) as f64;
        assert!(late_frac > early_frac + 0.12, "REM fraction rises across the night: early {early_frac:.2} → late {late_frac:.2}");

        // SLEEP DEBT: a night cut short (woken by the morning clock while still tired) leaves pressure high
        let mut tired = SleepCycle::new();
        tired.pressure = 0.95;
        tired.fall_asleep();
        for _ in 0..40 {
            tired.sleep_tick(0.02); // only ~0.8 h of sleep before...
        }
        // ...morning comes (clock ~08:00, low circadian sleepiness): it wakes, but still carrying debt
        assert!(tired.should_wake(0.34) == (tired.pressure < 0.5), "morning wake respects remaining sleep debt");
        assert!(tired.pressure > 0.7, "a night cut short leaves them still tired (sleep debt carried)");

        // (3) SHY DOWNSCALING: sleep prunes the trivial, keeps the salient, lowers the total
        let mut w = vec![0.9, 0.8, 0.15, 0.12, 0.1, 0.7];
        let (total_before, kept_before) = (w.iter().sum::<f64>(), w.len());
        downscale(&mut w, 0.6, 0.2);
        assert!(w.contains(&(0.9 * 0.6)) && w.contains(&(0.8 * 0.6)), "salient memories survive sleep");
        assert!(w.len() < kept_before && !w.iter().any(|&s| s < 0.2), "trivial traces pruned (forgotten)");
        assert!(w.iter().sum::<f64>() < total_before, "total synaptic strength renormalised DOWN");
        let weakest = w.iter().cloned().fold(f64::MAX, f64::min);
        assert!(weakest > 0.15, "the faintest surviving memory outranks the forgotten noise: {weakest:.2}");
    }
}
