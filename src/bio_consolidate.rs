//! bio_consolidate (memory #4, KEYSTONE): synaptic tagging & capture (Frey–Morris) — early→late LTP,
//! protein-gated. Weak-alone forgotten; weak+strong captured; protein-block forgets; salience(the neuromodulator)
//! rescues (flashbulb). Ported from bio_consolidate.py.

const PRP_THRESHOLD: f64 = 0.6;

#[derive(Clone)]
pub struct ConsolidatingSynapse {
    e: f64,
    l: f64,
    tag: f64,
}
impl ConsolidatingSynapse {
    fn new() -> Self {
        ConsolidatingSynapse { e: 0.0, l: 0.0, tag: 0.0 }
    }
    fn stimulate(&mut self, strength: f64) {
        self.e += strength;
        self.tag = self.tag.max(strength.min(1.0));
    }
    fn consolidate_step(&mut self, prp: f64) {
        self.l += 0.6 * self.tag * prp; // tag + PRP → permanent late-LTP
        self.e *= 0.85;
        self.tag *= 0.80;
    }
    pub fn weight(&self) -> f64 {
        self.e + self.l
    }
}

pub struct Cell {
    pub syn: Vec<ConsolidatingSynapse>,
    prp: f64,
    protein_synthesis: bool,
}
impl Cell {
    pub fn new(n: usize, protein_synthesis: bool) -> Self {
        Cell { syn: vec![ConsolidatingSynapse::new(); n], prp: 0.0, protein_synthesis }
    }
    pub fn event(&mut self, i: usize, strength: f64, salience: f64) {
        self.syn[i].stimulate(strength);
        let trigger = strength.max(salience);
        if self.protein_synthesis && trigger >= PRP_THRESHOLD {
            self.prp = (self.prp + trigger).min(1.0);
        }
    }
    pub fn wait(&mut self, steps: usize) {
        for _ in 0..steps {
            for s in self.syn.iter_mut() {
                s.consolidate_step(self.prp);
            }
            self.prp *= 0.85;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const REMEMBERED: f64 = 0.2;

    #[test]
    fn tag_and_capture() {
        // 1. weak alone → forgotten
        let mut c = Cell::new(2, true);
        c.event(0, 0.4, 0.0);
        c.wait(40);
        assert!(c.syn[0].weight() < REMEMBERED, "weak alone forgotten");

        // 2. weak + strong nearby → the weak tag captures the shared PRP → consolidated
        let mut c = Cell::new(2, true);
        c.event(0, 0.4, 0.0);
        c.event(1, 1.0, 0.0);
        c.wait(40);
        assert!(c.syn[0].weight() >= REMEMBERED, "weak captured by a strong neighbour");

        // 3. strong but protein synthesis blocked → forgotten
        let mut c = Cell::new(2, false);
        c.event(0, 1.0, 0.0);
        c.wait(40);
        assert!(c.syn[0].weight() < REMEMBERED, "no protein → forgotten");

        // 4. weak but SALIENT (the neuromodulator) → protein synthesis rescues it (flashbulb)
        let mut c = Cell::new(2, true);
        c.event(0, 0.4, 0.9);
        c.wait(40);
        assert!(c.syn[0].weight() >= REMEMBERED, "salience rescues the weak memory");
    }
}
