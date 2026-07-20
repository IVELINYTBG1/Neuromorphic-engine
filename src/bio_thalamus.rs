//! bio_thalamus (sensing #4): the thalamus as an attention/expectation gate (not a relay) — top-down
//! gain + reticular searchlight + expectation prior, but modulatory not generative. From bio_thalamus.py.

use crate::vec::relu;

pub struct Thalamus {
    g_att: f64,
    g_exp: f64,
    thresh: f64,
    w_ret: f64,
}

impl Default for Thalamus {
    fn default() -> Self {
        Thalamus { g_att: 1.5, g_exp: 0.4, thresh: 0.7, w_ret: 1.0 }
    }
}

impl Thalamus {
    pub fn relay(&self, sensory: &[f64], attention: Option<&[f64]>, expectation: Option<&[f64]>) -> Vec<f64> {
        let n = sensory.len();
        let att: Vec<f64> = attention.map(|a| a.to_vec()).unwrap_or_else(|| vec![0.0; n]);
        let exp: Vec<f64> = expectation.map(|e| e.to_vec()).unwrap_or_else(|| vec![0.0; n]);
        let attended: Vec<f64> = (0..n).map(|i| att[i] * sensory[i]).collect();
        let total: f64 = attended.iter().sum();
        (0..n)
            .map(|i| {
                let reticular = total - attended[i];
                relu(sensory[i] * (1.0 + self.g_att * att[i]) + self.g_exp * exp[i] - self.thresh - self.w_ret * reticular)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_expectation_gate() {
        let th = Thalamus::default();

        // attentional gain: attended passes, unattended blocked
        let attended = th.relay(&[0.6], Some(&[1.0]), None)[0];
        let unattended = th.relay(&[0.6], Some(&[0.0]), None)[0];
        assert!(attended > 0.1 && unattended < 0.01, "attention amplifies");

        // modulatory, not generative — can't hallucinate
        assert!(th.relay(&[0.0], Some(&[1.0]), None)[0] < 1e-6, "no input → no output");

        // reticular searchlight: attend one → suppress the other
        let base = th.relay(&[0.9, 0.9], None, None);
        let sel = th.relay(&[0.9, 0.9], Some(&[1.0, 0.0]), None);
        assert!(sel[0] > base[0] && sel[1] < base[1] - 0.1, "searchlight suppresses the unattended");

        // expectation lowers the threshold (a prior lets weak input in)
        let expected = th.relay(&[0.5, 0.5], None, Some(&[1.0, 0.0]));
        assert!(expected[0] > 0.05 && expected[1] < 0.01, "expectation passes the expected");
    }
}
