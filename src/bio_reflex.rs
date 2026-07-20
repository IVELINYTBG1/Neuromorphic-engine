//! bio_reflex (motor #4): the monosynaptic stretch reflex — a short-latency spinal negative-feedback
//! loop that rejects loads; γ-gain tunes stiffness; long delay = instability. Ported from bio_reflex.py.

pub fn simulate(gain: f64, delay: usize) -> Vec<f64> {
    let (gain_v, steps, pert, pert_start, m, k_pass, b_pass, dt) =
        (1.0, 700usize, 1.0, 80usize, 1.0, 1.0, 0.4, 0.05);
    let (mut x, mut v) = (0.0f64, 0.0f64);
    let mut hist: Vec<(f64, f64)> = vec![];
    let mut out = vec![0.0; steps];
    for t in 0..steps {
        let f_ext = if t >= pert_start { pert } else { 0.0 };
        let (xs, vs) = if t >= delay { hist[t - delay] } else { (0.0, 0.0) };
        let f_reflex = -gain * xs - gain_v * vs;
        let a = (f_ext + f_reflex - k_pass * x - b_pass * v) / m;
        v += dt * a;
        x += dt * v;
        hist.push((x, v));
        out[t] = x;
    }
    out
}

pub fn peak(out: &[f64]) -> f64 {
    out[80..].iter().map(|x| x.abs()).fold(0.0, f64::max)
}
pub fn steady(out: &[f64]) -> f64 {
    let tail = &out[out.len() - 60..];
    tail.iter().map(|x| x.abs()).sum::<f64>() / tail.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_reflex_stabilizes() {
        let (reflex_delay, cortical_delay) = (2, 22);
        let intact = simulate(8.0, reflex_delay);
        assert!(steady(&intact) < 0.2 && peak(&intact) < 0.3, "rejects a step load");

        // γ-gain tunes stiffness — stiffer reflex, smaller deflection
        let d: Vec<f64> = [2.0, 8.0, 20.0].iter().map(|&g| steady(&simulate(g, reflex_delay))).collect();
        assert!(d[0] > d[1] && d[1] > d[2] && d[2] < 0.1, "gain tunable");

        // cut the loop → load wins
        assert!(steady(&simulate(0.0, reflex_delay)) > 4.0 * steady(&intact), "loop cut → knocked off");

        // long delay → instability (short latency is why the reflex is spinal)
        assert!(peak(&simulate(8.0, cortical_delay)) > 2.0 * peak(&intact), "delay = instability");
    }
}
