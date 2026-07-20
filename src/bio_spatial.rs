//! bio_spatial (sensing #8): the hippocampal-entorhinal cognitive map — place cells decode position, grid
//! cells give a multi-scale modular metric that path-integrates self-motion, and an allocentric map turns
//! reactive taxis into vector navigation. Ported from bio_spatial.py.

use crate::rng::Rng;
use crate::vec;
use std::f64::consts::PI;

pub struct PlaceMap {
    pub centers: Vec<(f64, f64)>,
    width: f64,
}

impl PlaceMap {
    pub fn new() -> Self {
        let (n_side, arena) = (9usize, 10.0);
        let xs: Vec<f64> = (0..n_side).map(|i| arena * i as f64 / (n_side - 1) as f64).collect();
        let mut centers = vec![];
        for &x in &xs {
            for &y in &xs {
                centers.push((x, y));
            }
        }
        PlaceMap { centers, width: 1.0 }
    }
    pub fn encode(&self, pos: (f64, f64)) -> Vec<f64> {
        self.centers.iter().map(|&(cx, cy)| {
            let d2 = (cx - pos.0).powi(2) + (cy - pos.1).powi(2);
            (-d2 / (2.0 * self.width * self.width)).exp()
        }).collect()
    }
    pub fn decode_with(&self, rates: &[f64], centers: &[(f64, f64)]) -> (f64, f64) {
        let s: f64 = rates.iter().sum::<f64>() + 1e-9;
        let mut xy = (0.0, 0.0);
        for i in 0..centers.len() {
            let w = rates[i] / s;
            xy.0 += w * centers[i].0;
            xy.1 += w * centers[i].1;
        }
        xy
    }
    pub fn decode(&self, rates: &[f64]) -> (f64, f64) {
        self.decode_with(rates, &self.centers)
    }
}

pub struct GridModules {
    periods: Vec<f64>,
    phase: Vec<f64>,
}

impl GridModules {
    pub fn new() -> Self {
        GridModules { periods: vec![3.0, 3.5, 4.0], phase: vec![0.0; 3] }
    }
    pub fn reset(&mut self, pos: f64) {
        self.phase = self.periods.iter().map(|&l| pos.rem_euclid(l)).collect();
    }
    pub fn step(&mut self, dx: f64) {
        for i in 0..self.periods.len() {
            self.phase[i] = (self.phase[i] + dx).rem_euclid(self.periods[i]);
        }
    }
    pub fn decode(&self, max_pos: f64, modules: Option<&[usize]>) -> f64 {
        let idx: Vec<usize> = match modules {
            Some(m) => m.to_vec(),
            None => (0..self.periods.len()).collect(),
        };
        let (mut best, mut best_cost, mut p) = (0.0, f64::INFINITY, 0.0);
        while p < max_pos {
            let cost: f64 = idx.iter().map(|&i| {
                let (l, ph) = (self.periods[i], self.phase[i]);
                (p - ph).rem_euclid(l).min((ph - p).rem_euclid(l))
            }).sum();
            if cost < best_cost {
                best_cost = cost;
                best = p;
            }
            p += 0.02;
        }
        best
    }
}

fn ang(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % (2.0 * PI);
    d.min(2.0 * PI - d)
}
fn direction(v: (f64, f64)) -> f64 {
    v.1.atan2(v.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognitive_map() {
        let mut g = Rng::new(0);
        let pm = PlaceMap::new();

        // place cells decode allocentric position
        let pts = [(2.5, 3.0), (5.0, 5.0), (7.0, 4.5), (3.5, 7.0), (6.0, 6.5)];
        let place_err = pts.iter().map(|&p| {
            let d = pm.decode(&pm.encode(p));
            ((d.0 - p.0).powi(2) + (d.1 - p.1).powi(2)).sqrt()
        }).fold(0.0, f64::max);
        assert!(place_err < 0.7, "place cells decode position");

        // grid cells: multi-scale uniquely localizes where a single module aliases
        let mut gc = GridModules::new();
        gc.reset(17.3);
        assert!((gc.decode(42.0, None) - 17.3).abs() < 0.3 && (gc.decode(42.0, Some(&[0])) - 17.3).abs() > 2.0,
                "grid modules resolve aliasing");

        // path integration: dead-reckon, drift, landmark correct
        let mut gc2 = GridModules::new();
        gc2.reset(0.0);
        for _ in 0..30 {
            gc2.step(0.4);
        }
        assert!((gc2.decode(42.0, None) - 12.0).abs() < 0.3, "dead reckoning");
        let mut gc3 = GridModules::new();
        gc3.reset(0.0);
        let mut pos = 0.0;
        for _ in 0..30 {
            pos += 0.4;
            gc3.step(0.4 + 0.12 * g.normal());
        }
        let drift = (gc3.decode(42.0, None) - pos).abs();
        gc3.reset(pos);
        assert!(drift > 0.3 && (gc3.decode(42.0, None) - pos).abs() < 0.15, "drift then landmark-corrected");

        // vector navigation: allocentric map → goal direction from a novel start; no map = chance
        let goal = (8.0, 8.0);
        let starts: Vec<(f64, f64)> = (0..24).map(|_| (g.uniform() * 7.0 + 1.0, g.uniform() * 7.0 + 1.0)).collect();
        let true_dir: Vec<f64> = starts.iter().map(|&s| direction((goal.0 - s.0, goal.1 - s.1))).collect();
        let map_err = starts.iter().zip(&true_dir).map(|(&s, &td)| {
            let d = pm.decode(&pm.encode(s));
            ang(direction((goal.0 - d.0, goal.1 - d.1)), td)
        }).sum::<f64>() / starts.len() as f64;
        let chance_err = true_dir.iter().map(|&td| ang(g.uniform() * 2.0 * PI, td)).sum::<f64>() / true_dir.len() as f64;
        assert!(map_err < 0.2 && chance_err > 1.0, "map navigates; no map is chance");

        // remapping: a new environment recruits an orthogonal place code
        let perm = g.randperm(pm.centers.len());
        let centers_b: Vec<(f64, f64)> = perm.iter().map(|&i| pm.centers[i]).collect();
        let encode_b = |pos: (f64, f64)| -> Vec<f64> {
            centers_b.iter().map(|&(cx, cy)| (-((cx - pos.0).powi(2) + (cy - pos.1).powi(2)) / 2.0).exp()).collect()
        };
        let sample: Vec<(f64, f64)> = (0..40).map(|_| (g.uniform() * 10.0, g.uniform() * 10.0)).collect();
        let cross = sample.iter().map(|&p| vec::pearson(&pm.encode(p), &encode_b(p))).sum::<f64>() / sample.len() as f64;
        assert!(cross < 0.3, "orthogonal maps per environment");
    }
}
