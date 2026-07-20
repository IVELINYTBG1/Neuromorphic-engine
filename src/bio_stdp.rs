//! bio_stdp (learning #5): the Bi-Poo STDP window — pre-before-post → LTP, post-before-pre → LTD, fading
//! with |Δt|. Pre/post eligibility traces; the depression gate is the PRE spike, potentiation the POST.
//! Ported from bio_stdp.py.

pub struct BioSynapse {
    dpre: f64,
    dpost: f64,
    a_plus: f64,
    a_minus: f64,
    pre_tr: f64,
    post_tr: f64,
}

impl Default for BioSynapse {
    fn default() -> Self {
        BioSynapse {
            dpre: (-1.0f64 / 20.0).exp(),
            dpost: (-1.0f64 / 20.0).exp(),
            a_plus: 0.10,
            a_minus: 0.105,
            pre_tr: 0.0,
            post_tr: 0.0,
        }
    }
}

impl BioSynapse {
    pub fn reset_state(&mut self) {
        self.pre_tr = 0.0;
        self.post_tr = 0.0;
    }
    pub fn step(&mut self, pre_spk: f64, post_spk: f64) -> f64 {
        self.pre_tr *= self.dpre;
        self.post_tr *= self.dpost;
        let mut dw = -self.a_minus * pre_spk * self.post_tr; // pre after post → LTD
        dw += self.a_plus * post_spk * self.pre_tr; // post after pre → LTP
        self.pre_tr += pre_spk;
        self.post_tr += post_spk;
        dw
    }
}

/// Total Δw for a single pre/post pair with timing Δt = t_post − t_pre.
pub fn stdp_window(syn: &mut BioSynapse, dt: i32) -> f64 {
    let gap = 40;
    syn.reset_state();
    let (t_pre, t_post) = (gap, gap + dt);
    let horizon = gap + dt.abs() + 1;
    let mut total = 0.0;
    for t in 0..horizon {
        total += syn.step((t == t_pre) as i32 as f64, (t == t_post) as i32 as f64);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bi_poo_window() {
        let mut syn = BioSynapse::default();
        let dw: Vec<(i32, f64)> = (-25..=25).map(|dt| (dt, stdp_window(&mut syn, dt))).collect();
        let get = |dt: i32| dw.iter().find(|(d, _)| *d == dt).unwrap().1;

        // causal (pre before post, Δt>0) → LTP; anticausal (Δt<0) → LTD
        assert!((1..=25).all(|dt| get(dt) > 0.0), "causal → LTP");
        assert!((-25..0).all(|dt| get(dt) < 0.0), "anticausal → LTD");

        // fades with |Δt|
        assert!(get(2) > get(20) && get(-2) < get(-20), "window fades with |Δt|");
    }
}
