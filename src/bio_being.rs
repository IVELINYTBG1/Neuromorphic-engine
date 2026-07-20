//! bio_being (CAPSTONE II): the LATER families wired into one being — cognitive map (grown pathways),
//! body→feeling (insula), hormones (cortisol), glia (D-serine gates memory), structural memory. Each is
//! load-bearing: map beats taxis, feeling protects, glia forms the memory the map needs. From bio_being.py.

use crate::bio_glia::Astrocyte;
use crate::bio_interoception::{Insula, SETPOINTS};
use crate::bio_motorcortex::MotorCortex;
use crate::bio_neuroendocrine::{Neuroendocrine, EXPLORATION, PLASTICITY, THREAT_GAIN, VIGOR};
use crate::bio_select::BasalGanglia;
use crate::bio_structural::StructuralNet;
use crate::rng::Rng;
use crate::vec;
use std::f64::consts::PI;

const N_ACT: usize = 8;
const ARENA: f64 = 12.0;
const STEP: f64 = 0.6;
const SENSE_RANGE: f64 = 2.0;
const GRID: usize = 6;
const FOOD_TOKEN: usize = GRID * GRID;
const THREAT_TOKEN: usize = GRID * GRID + 1;
const REGROW: i32 = 28;
const FOOD_R: f64 = 0.8;
const THREAT_R: f64 = 1.1;
const FOOD_SITES: [(f64, f64); 2] = [(1.5, 6.0), (10.5, 6.0)];
const THREAT: (f64, f64) = (6.0, 6.0);

fn angle(a: usize) -> f64 {
    2.0 * PI * a as f64 / N_ACT as f64
}
fn pbin(p: (f64, f64)) -> usize {
    let ix = ((p.0 / ARENA * GRID as f64) as usize).min(GRID - 1);
    let iy = ((p.1 / ARENA * GRID as f64) as usize).min(GRID - 1);
    iy * GRID + ix
}
fn bin_center(b: usize) -> (f64, f64) {
    let (ix, iy) = (b % GRID, b / GRID);
    ((ix as f64 + 0.5) * ARENA / GRID as f64, (iy as f64 + 0.5) * ARENA / GRID as f64)
}
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

pub struct Being {
    gen: Rng,
    insula: Insula,
    ne: Neuroendocrine,
    astro: Astrocyte,
    mem: StructuralNet,
    bg: BasalGanglia,
    mc: MotorCortex,
    q: Vec<f64>,
    use_map: bool,
    use_intero: bool,
    use_glia: bool,
    pos: (f64, f64),
    energy: f64,
    site_timer: [i32; 2],
    pub food_eaten: usize,
    pub threat_hits: usize,
}

impl Being {
    pub fn new(seed: u64, use_map: bool, use_intero: bool, use_glia: bool) -> Self {
        Being {
            gen: Rng::new(seed),
            insula: Insula::new(SETPOINTS, 1.0),
            ne: Neuroendocrine::new(),
            astro: Astrocyte::default_(),
            mem: StructuralNet::new(2, 0.0, true, vec![]),
            bg: BasalGanglia::new(50),
            mc: MotorCortex::new(seed),
            q: vec![0.0; GRID * GRID],
            use_map, use_intero, use_glia,
            pos: (6.0, 1.0),
            energy: 1.0,
            site_timer: [0, 0],
            food_eaten: 0,
            threat_hits: 0,
        }
    }

    fn sensed_food(&self) -> Option<usize> {
        let (mut best, mut bd) = (None, 1e9);
        for (i, &s) in FOOD_SITES.iter().enumerate() {
            if self.site_timer[i] == 0 {
                let d = dist(self.pos, s);
                if d < SENSE_RANGE && d < bd {
                    best = Some(i);
                    bd = d;
                }
            }
        }
        best
    }
    fn remembered_food(&self) -> Option<(f64, f64)> {
        if !self.use_map {
            return None;
        }
        let mut bins: Vec<usize> = self.mem.syn.keys().filter(|&&(_, post)| post == FOOD_TOKEN).map(|&(pre, _)| pre).collect();
        bins.sort();
        bins.dedup();
        let (mut best, mut bv) = (None, f64::NEG_INFINITY);
        for b in bins {
            let c = bin_center(b);
            if dist(c, self.pos) < 1.0 {
                continue;
            }
            let v = self.q[b] - 0.02 * dist(c, self.pos);
            if v > bv {
                bv = v;
                best = Some(c);
            }
        }
        best
    }

    pub fn step(&mut self) {
        let threat_d = dist(self.pos, THREAT);
        let threat_prox = (-threat_d / 1.4).exp();
        let hunger = 1.0 - self.energy;
        let body = [1.0 + 1.3 * threat_prox, 1.0 + 0.8 * threat_prox, self.energy, 1.0];
        let (_valence, arousal) = if self.use_intero {
            self.insula.feel(&body, -0.6 * threat_prox)
        } else {
            (-hunger, 0.0)
        };
        if self.use_intero {
            self.ne.release("cortisol", 2.5 * arousal * threat_prox);
        }
        self.ne.tick(0.0);
        let modu = self.ne.modulation();
        let threat_gain = modu[THREAT_GAIN];
        let explore_t = modu[EXPLORATION].max(0.2);
        self.astro.integrate(if self.use_glia { arousal } else { 0.0 }, 0.0);
        let d_serine = if self.use_glia { self.astro.d_serine() } else { 0.0 };

        let sensed = self.sensed_food();
        let goal = match sensed {
            Some(i) => Some(FOOD_SITES[i]),
            None => self.remembered_food(),
        };
        let mut sal = vec![0.0; N_ACT];
        if let Some(g) = goal {
            let gdir = (g.1 - self.pos.1).atan2(g.0 - self.pos.0);
            for a in 0..N_ACT {
                sal[a] += 1.5 * (angle(a) - gdir).cos().max(0.0);
            }
        }
        let away = (self.pos.1 - THREAT.1).atan2(self.pos.0 - THREAT.0);
        let fear = arousal * threat_gain * threat_prox;
        for a in 0..N_ACT {
            sal[a] += 3.4 * fear * (angle(a) - away).cos().max(0.0);
        }
        for a in 0..N_ACT {
            sal[a] = vec::relu(sal[a] + 0.25 * explore_t * self.gen.normal());
        }
        let (r_act, selm) = self.bg.select(&sal, 1.2);
        let a = if selm.iter().any(|&x| x) { vec::argmax(&r_act) } else { self.gen.randint(N_ACT) };

        let (pv, _) = self.mc.decode(angle(a), None);
        let vigor = modu[VIGOR].clamp(0.6, 1.3);
        self.pos.0 = (self.pos.0 + STEP * vigor * pv.cos()).clamp(0.0, ARENA);
        self.pos.1 = (self.pos.1 + STEP * vigor * pv.sin()).clamp(0.0, ARENA);

        self.energy = (self.energy - 0.012).max(0.0);
        let mut ate = false;
        for i in 0..FOOD_SITES.len() {
            if self.site_timer[i] == 0 && dist(self.pos, FOOD_SITES[i]) < FOOD_R {
                self.energy = (self.energy + 0.5).min(1.0);
                self.site_timer[i] = REGROW;
                self.food_eaten += 1;
                ate = true;
                if self.use_intero {
                    self.ne.release("oxytocin", 0.6);
                }
            }
        }
        let hit = dist(self.pos, THREAT) < THREAT_R;
        if hit {
            self.energy = (self.energy - 0.15).max(0.0);
            self.threat_hits += 1;
        }
        for i in 0..FOOD_SITES.len() {
            if self.site_timer[i] > 0 {
                self.site_timer[i] -= 1;
            }
        }

        let gate_open = d_serine > 0.15 || (self.use_glia && (ate || hit) && arousal > 0.05);
        if ate && self.use_glia && gate_open {
            let b = pbin(self.pos);
            for _ in 0..3 {
                self.mem.expose(b, FOOD_TOKEN);
            }
        }
        if hit && self.use_glia && gate_open {
            let b = pbin(self.pos);
            for _ in 0..3 {
                self.mem.expose(b, THREAT_TOKEN);
            }
        }
        if ate {
            let reward = 0.99; // -0.01 step cost + 1.0 food
            let lr = 0.4 * modu[PLASTICITY] * if !self.use_glia { 1.0 } else { (d_serine + 0.2).max(0.1) };
            let b = pbin(self.pos);
            self.q[b] += lr * (reward - self.q[b]);
        }
    }

    fn n_remembered(&self) -> usize {
        let mut bins: Vec<usize> = self.mem.syn.keys().filter(|&&(_, post)| post == FOOD_TOKEN).map(|&(pre, _)| pre).collect();
        bins.sort();
        bins.dedup();
        bins.len()
    }
}

/// returns (food_eaten, threat_hits, n_remembered_food_sites)
pub fn run(steps: usize, seed: u64, use_map: bool, use_intero: bool, use_glia: bool) -> (usize, usize, usize) {
    let mut b = Being::new(seed, use_map, use_intero, use_glia);
    for _ in 0..steps {
        b.step();
    }
    (b.food_eaten, b.threat_hits, b.n_remembered())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_later_families_compose_into_one_being() {
        let steps = 1500;
        let (food_full, hits_full, remembered_full) = run(steps, 0, true, true, true);
        let (food_nomap, _, _) = run(steps, 0, false, true, true);
        let (_, hits_nointero, _) = run(steps, 0, true, false, true);
        let (food_noglia, _, remembered_noglia) = run(steps, 0, true, true, false);

        // the cognitive map pays off (navigate to remembered food out of sight, beats taxis)
        assert!(food_full > (1.4 * food_nomap as f64) as usize && remembered_full >= 2,
                "map navigates ({} vs no-map {})", food_full, food_nomap);
        // body→feeling→cortisol makes it avoid the threat (deafen insula → walks into danger)
        assert!(hits_nointero as f64 > 1.5 * hits_full as f64 + 2.0,
                "feeling protects ({} hits with vs {} without interoception)", hits_full, hits_nointero);
        // glia D-serine gates the spatial memory the map is built from
        assert!(remembered_noglia == 0 && (food_noglia as f64) < 1.4 * food_nomap as f64,
                "no glia → no memory → no map");
    }
}
