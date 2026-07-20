//! bio_structural (memory #9): memory as STRUCTURAL plasticity — synaptogenesis (new edges), pruning
//! (use it or lose it), neurogenesis (recruit fresh cells). The topology itself is the memory. An
//! explicit sparse edge set, not a dense matrix. Ported from bio_structural.py.

use std::collections::HashMap;

pub struct StructuralNet {
    pub syn: HashMap<(usize, usize), f64>,
    contact: HashMap<(usize, usize), usize>,
    contact_thresh: usize,
    w_init: f64,
    w_max: f64,
    prune_thresh: f64,
    hebb: f64,
    decay: f64,
    synaptogenesis: bool,
    pool: Vec<usize>,
    pub committed: Vec<usize>,
}

impl StructuralNet {
    pub fn new(contact_thresh: usize, decay: f64, synaptogenesis: bool, pool: Vec<usize>) -> Self {
        StructuralNet {
            syn: HashMap::new(),
            contact: HashMap::new(),
            contact_thresh,
            w_init: 0.3,
            w_max: 1.0,
            prune_thresh: 0.05,
            hebb: 0.3,
            decay,
            synaptogenesis,
            pool,
            committed: vec![],
        }
    }
    pub fn expose(&mut self, pre: usize, post: usize) {
        let key = (pre, post);
        if let Some(w) = self.syn.get_mut(&key) {
            *w = (*w + self.hebb).min(self.w_max); // Hebbian on the existing edge
        } else if self.synaptogenesis {
            let c = self.contact.entry(key).or_insert(0);
            *c += 1;
            if *c >= self.contact_thresh {
                self.syn.insert(key, self.w_init); // SYNAPTOGENESIS — a new pathway
                self.contact.remove(&key);
            }
        }
    }
    pub fn rest(&mut self) {
        let mut dead = vec![];
        for (k, w) in self.syn.iter_mut() {
            *w -= self.decay;
            if *w < self.prune_thresh {
                dead.push(*k);
            }
        }
        for k in dead {
            self.syn.remove(&k); // PRUNING
        }
    }
    pub fn recall(&self, pre: usize) -> Option<usize> {
        let mut out: HashMap<usize, f64> = HashMap::new();
        for (&(p, q), &w) in self.syn.iter() {
            if p == pre {
                *out.entry(q).or_insert(0.0) += w;
            }
        }
        out.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(q, _)| q)
    }
    pub fn recruit(&mut self) -> usize {
        let t = self.pool.remove(0);
        self.committed.push(t);
        t
    }
    pub fn store_novel(&mut self, cue: usize) -> usize {
        let target = self.recruit();
        for _ in 0..self.contact_thresh + 1 {
            self.expose(cue, target);
        }
        target
    }
    pub fn n_synapses(&self) -> usize {
        self.syn.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_as_new_pathways() {
        // synaptogenesis: 0 edges → grow 3 pathways that recall (decay 0.02 so unused edges prune)
        let mut net = StructuralNet::new(3, 0.02, true, vec![]);
        assert_eq!(net.n_synapses(), 0);
        let pairs = [(0, 8), (1, 9), (2, 10)];
        for _ in 0..4 {
            for &(a, b) in pairs.iter() {
                net.expose(a, b);
            }
        }
        assert_eq!(net.n_synapses(), 3);
        assert!(pairs.iter().all(|&(a, b)| net.recall(a) == Some(b)), "grown pathways recall");

        // pruning: keep two, abandon the third → it is removed
        for _ in 0..40 {
            net.expose(1, 9);
            net.expose(2, 10);
            net.rest();
        }
        assert!(net.recall(0).is_none() && net.recall(1) == Some(9) && net.n_synapses() == 2,
                "abandoned pathway pruned, rehearsed persist");

        // topology IS the memory: |edges| tracks |memories|
        let mut net2 = StructuralNet::new(3, 0.0, true, vec![]);
        let mut counts = vec![];
        for &(pre, post) in &[(0, 5), (1, 6), (2, 7), (3, 4)] {
            for _ in 0..4 {
                net2.expose(pre, post);
            }
            counts.push(net2.n_synapses());
        }
        assert_eq!(counts, vec![1, 2, 3, 4]);

        // neurogenesis: novel cues recruit distinct fresh neurons
        let mut net3 = StructuralNet::new(3, 0.0, true, (8..16).collect());
        let (ta, tb, tc) = (net3.store_novel(0), net3.store_novel(1), net3.store_novel(2));
        assert!(ta != tb && tb != tc && ta != tc && net3.recall(0) == Some(ta) && net3.committed.len() == 3);

        // teeth: no synaptogenesis → nothing is ever stored
        let mut fixed = StructuralNet::new(3, 0.0, false, vec![]);
        for _ in 0..10 {
            for &(a, b) in pairs.iter() {
                fixed.expose(a, b);
            }
        }
        assert!(fixed.n_synapses() == 0 && pairs.iter().all(|&(a, _)| fixed.recall(a).is_none()),
                "structure is what stores");
    }
}
