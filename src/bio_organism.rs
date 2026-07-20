//! bio_organism (CAPSTONE I): the five core families wired into one closed sensorimotor foraging loop —
//! sense → feel → select → act → learn (RPE) → remember. Ablate the value loop and foraging collapses to
//! a random walk. Composes the tested cells, no backprop. Ported from bio_organism.py.

use crate::bio_motorcortex::MotorCortex;
use crate::bio_neuromod::NeuroModulator;
use crate::bio_phill::SalienceGatedMemory;
use crate::bio_select::BasalGanglia;
use crate::rng::Rng;
use crate::vec;
use std::f64::consts::PI;

const N_ACT: usize = 8;
const WORLD: f64 = 10.0;
const STEP: f64 = 0.5;
const FOOD_R: f64 = 0.7;
const THREAT_R: f64 = 0.7;
const THREAT_SCALE: f64 = 1.6;

fn angle(a: usize) -> f64 {
    2.0 * PI * a as f64 / N_ACT as f64
}
fn dir_bin(a: f64) -> usize {
    ((a.rem_euclid(2.0 * PI)) / (2.0 * PI) * N_ACT as f64) as usize % N_ACT
}

pub struct Organism {
    gen: Rng,
    nm: NeuroModulator,
    bg: BasalGanglia,
    mc: MotorCortex,
    mem: SalienceGatedMemory,
    q: Vec<Vec<f64>>, // N_ACT contexts × N_ACT actions
    learn: bool,
    avoid: bool,
    flat_memory: bool,
    pos: (f64, f64),
    food: (f64, f64),
    threat: (f64, f64),
    pub exposures: Vec<(usize, usize, f64)>,
}

impl Organism {
    pub fn new(seed: u64, learn: bool, avoid: bool, flat_memory: bool) -> Self {
        let mut gen = Rng::new(seed);
        let pos = (gen.uniform() * WORLD, gen.uniform() * WORLD);
        let food = (gen.uniform() * WORLD, gen.uniform() * WORLD);
        Organism {
            gen,
            nm: NeuroModulator { base_lr: 0.15, base_gain: 2.0, base_temp: 0.25, k: 3.0 },
            bg: BasalGanglia::new(60),
            mc: MotorCortex::new(seed),
            mem: SalienceGatedMemory::new(N_ACT, 0.15),
            q: vec![vec![0.0; N_ACT]; N_ACT],
            learn, avoid, flat_memory,
            pos, food,
            threat: (WORLD / 2.0, WORLD / 2.0),
            exposures: vec![],
        }
    }

    pub fn step(&mut self) -> (bool, bool) {
        let vf = (self.food.0 - self.pos.0, self.food.1 - self.pos.1);
        let dist_f = (vf.0 * vf.0 + vf.1 * vf.1).sqrt();
        let ctx = dir_bin(vf.1.atan2(vf.0));
        let vt = (self.pos.0 - self.threat.0, self.pos.1 - self.threat.1);
        let dist_t = (vt.0 * vt.0 + vt.1 * vt.1).sqrt();
        let phi_away = vt.1.atan2(vt.0);

        let threat_prox = (-dist_t / THREAT_SCALE).exp();
        let m = (0.15 + threat_prox).min(1.0);

        let mut sal: Vec<f64> = self.q[ctx].iter().map(|&v| vec::relu(v)).collect();
        if self.avoid {
            for a in 0..N_ACT {
                sal[a] += 1.2 * threat_prox * (angle(a) - phi_away).cos().max(0.0);
            }
        }
        for a in 0..N_ACT {
            sal[a] = vec::relu(sal[a] + self.nm.temperature(m) * self.gen.normal());
        }
        let (r_act, sel) = self.bg.select(&sal, 1.2);
        let a = if sel.iter().any(|&x| x) { vec::argmax(&r_act) } else { vec::argmax(&self.q[ctx]) };

        let (pv, _) = self.mc.decode(angle(a), None);
        self.pos.0 = (self.pos.0 + STEP * pv.cos()).clamp(0.0, WORLD);
        self.pos.1 = (self.pos.1 + STEP * pv.sin()).clamp(0.0, WORLD);

        let new_dist_f = ((self.food.0 - self.pos.0).powi(2) + (self.food.1 - self.pos.1).powi(2)).sqrt();
        let mut r = dist_f - new_dist_f;
        let caught = new_dist_f < FOOD_R;
        let hit = ((self.pos.0 - self.threat.0).powi(2) + (self.pos.1 - self.threat.1).powi(2)).sqrt() < THREAT_R;
        if caught {
            r += 1.0;
            self.food = (self.gen.uniform() * WORLD, self.gen.uniform() * WORLD);
        }
        if hit {
            r -= 1.0;
        }
        if self.learn {
            self.q[ctx][a] += self.nm.lr(m) * (r - self.q[ctx][a]);
        }
        let felt = (m + 0.6 * (caught || hit) as i32 as f64).min(1.0);
        if caught || hit || threat_prox > 0.4 {
            let imprint = if self.flat_memory { 0.1 } else { felt };
            self.mem.expose(ctx, a, imprint);
            self.exposures.push((ctx, a, felt));
        }
        (caught, hit)
    }
}

pub fn run(steps: usize, seed: u64, learn: bool, avoid: bool, flat_memory: bool) -> (Organism, usize, usize) {
    let mut org = Organism::new(seed, learn, avoid, flat_memory);
    let (mut food, mut hits) = (0, 0);
    for _ in 0..steps {
        let (c, h) = org.step();
        food += c as usize;
        hits += h as usize;
    }
    (org, food, hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_families_compose_into_a_forager() {
        let steps = 2500;
        let (org, reaches, hits) = run(steps, 0, true, true, false);
        let (_, reaches_norpe, _) = run(steps, 0, false, true, false);
        let (_, _, hits_noavoid) = run(steps, 0, true, false, false);
        let (flat, _, _) = run(steps, 0, true, true, true);

        // learns to forage; ablate the RPE value loop → collapses to a random walk
        assert!(reaches > 15 && reaches > 3 * reaches_norpe, "forages ({} vs no-RPE {})", reaches, reaches_norpe);
        // reactive limbic avoidance keeps it off the threat
        assert!(hits < hits_noavoid, "avoidance reduces threat hits ({} vs {})", hits, hits_noavoid);

        // remembers what it FELT: felt memory > flat-affect twin across the salient episodes
        let sal_ex: Vec<(usize, usize)> = org.exposures.iter().map(|&(c, a, _)| (c, a)).collect();
        let felt = sal_ex.iter().map(|&(c, a)| org.mem.confidence(c, a)).sum::<f64>() / sal_ex.len().max(1) as f64;
        let flat_c = sal_ex.iter().map(|&(c, a)| flat.mem.confidence(c, a)).sum::<f64>() / sal_ex.len().max(1) as f64;
        assert!(felt > flat_c, "felt memory stronger than flat-affect ({:.3} vs {:.3})", felt, flat_c);
    }
}
