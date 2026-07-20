//! bio_dopamine (limbic #1): dopamine = temporal-difference reward-prediction-error (Schultz/Montague),
//! not reward. Learns value locally, shifts to the predictive cue, dips on omission. From bio_dopamine.py.

pub struct TDValue {
    pub v: Vec<f64>,
    alpha: f64,
    gamma: f64,
}

impl TDValue {
    pub fn new(n_states: usize) -> Self {
        TDValue { v: vec![0.0; n_states], alpha: 0.3, gamma: 0.95 }
    }
    pub fn step(&mut self, s: usize, r: f64, s_next: Option<usize>, learn: bool) -> f64 {
        let v_next = match s_next { Some(n) => self.v[n], None => 0.0 };
        let delta = r + self.gamma * v_next - self.v[s];
        if learn {
            self.v[s] += self.alpha * delta;
        }
        delta
    }
    /// State 0 is the ITI/baseline whose value is NOT learned (trial onset unpredictable).
    pub fn run_trial(&mut self, rewards: &[f64]) -> Vec<f64> {
        let t = rewards.len();
        let mut deltas = vec![0.0; t];
        for s in 0..t {
            let s_next = if s + 1 < t { Some(s + 1) } else { None };
            deltas[s] = self.step(s, rewards[s], s_next, s != 0);
        }
        deltas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dopamine_is_prediction_error() {
        let (t, cue, reward_t) = (10usize, 1usize, 8usize);
        let mut rewards = vec![0.0; t];
        rewards[reward_t] = 1.0;

        let mut td = TDValue::new(t);
        let first = td.run_trial(&rewards);
        let da_reward_first = first[reward_t];
        let da_cue_first = first[0];
        let mut last = first;
        for _ in 0..150 {
            last = td.run_trial(&rewards);
        }
        let da_reward_last = last[reward_t];
        let da_cue_last = last[0];
        let v_learned = td.v.clone();
        let da_omit = td.run_trial(&vec![0.0; t])[reward_t];

        assert!(da_reward_first > 0.5, "unexpected reward → burst");
        assert!(da_cue_last > da_cue_first + 0.2 && da_reward_last < 0.3 * da_reward_first,
                "response shifts from reward to cue");
        assert!(da_omit < -0.3, "omitted expected reward → dip");
        let backs_up = v_learned[cue] > 0.3
            && (cue..reward_t).all(|i| v_learned[i + 1] - v_learned[i] >= -1e-3);
        assert!(backs_up, "value backs up from reward toward the cue");
    }
}
