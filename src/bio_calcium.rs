//! bio_calcium (memory #2): calcium-control — one Ca²⁺ signal → none/LTD/LTP, and STDP timing DERIVED
//! from it (NMDA Ca²⁺ = glutamate AND depolarization; bAP fast, glutamate slow). From bio_calcium.py.

use crate::bio_nmda::mg_unblock;

const THETA_D: f64 = 0.15;
const THETA_P: f64 = 0.55;

pub fn omega(ca: f64) -> f64 {
    if ca >= THETA_P {
        ca - THETA_P // high Ca²⁺ → LTP
    } else if ca >= THETA_D {
        -(ca - THETA_D) // mid Ca²⁺ → LTD
    } else {
        0.0
    }
}

/// Ca²⁺ from spike timing (Δt = t_post − t_pre), normalized so Δt=0 = 1.0.
pub fn ca_from_timing(dt: i32) -> f64 {
    let (tau_glu, tau_bap, horizon) = (25.0_f64, 2.7_f64, 80i32);
    let integral = |t_pre: i32, t_post: i32| -> f64 {
        let (mut g, mut d, mut total) = (0.0, 0.0, 0.0);
        for t in 0..horizon {
            g *= (-1.0 / tau_glu).exp();
            d *= (-1.0 / tau_bap).exp();
            if t == t_pre { g += 1.0; }
            if t == t_post { d += 1.0; }
            total += g * d;
        }
        total
    };
    let base = 20;
    integral(base, base + dt) / integral(base, base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_calcium_signal_unifies_plasticity() {
        // Ω bands: low → none, mid → LTD, high → LTP
        assert!(omega(0.08) == 0.0 && omega(0.32) < 0.0 && omega(0.80) > 0.0);

        // BCM depolarization curve derived through the Mg²⁺ unblock
        assert!(omega(mg_unblock(0.15)) == 0.0);
        assert!(omega(mg_unblock(0.35)) < 0.0);
        assert!(omega(mg_unblock(1.0)) > 0.0);

        // STDP emerges: central causal (Δt>0) → LTP, anticausal (Δt<0) → LTD
        assert!([2, 4, 8].iter().all(|&dt| omega(ca_from_timing(dt)) > 0.0));
        assert!([-2, -4].iter().all(|&dt| omega(ca_from_timing(dt)) < 0.0));
    }
}
