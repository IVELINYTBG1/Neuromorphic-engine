//! bio_glia (glia #1): astrocytes are not glue — K⁺ buffering (stability), slow Ca²⁺ waves (a second
//! spatial signal), D-serine gating of plasticity (a necessary third factor), lactate fuelling firing.
//! Ported from bio_glia.py.

pub struct Astrocyte {
    k_uptake: f64,
    ca_decay: f64,
    ca_couple: f64,
    lactate_gain: f64,
    dt: f64,
    pub ca: f64,
}

impl Astrocyte {
    pub fn new(k_uptake: f64, ca_decay: f64, ca_couple: f64, lactate_gain: f64, dt: f64) -> Self {
        Astrocyte { k_uptake, ca_decay, ca_couple, lactate_gain, dt, ca: 0.0 }
    }
    pub fn default_() -> Self {
        Astrocyte::new(2.0, 0.08, 0.4, 0.07, 0.2)
    }
    pub fn buffer(&self, k_extra: f64) -> f64 {
        self.k_uptake * k_extra
    }
    pub fn integrate(&mut self, neural_input: f64, laplacian: f64) -> f64 {
        self.ca += self.dt * (neural_input + self.ca_couple * laplacian - self.ca_decay * self.ca);
        self.ca
    }
    pub fn d_serine(&self) -> f64 {
        self.ca.min(1.0)
    }
    pub fn lactate(&self, activity: f64) -> f64 {
        self.lactate_gain * activity
    }
}

pub fn potassium_stability(glia: bool) -> f64 {
    let (drive, gain, gamma, release, leak, dt): (f64, f64, f64, f64, f64, f64) =
        (0.5, 4.0, 2.0, 0.5, 0.05, 0.1);
    let astro = Astrocyte::new(2.0, 0.08, 0.4, 0.07, 0.2);
    let (mut r, mut k): (f64, f64) = (0.0, 0.0);
    for _ in 0..250 {
        r = 1.0 / (1.0 + (-gain * (drive + gamma * k - 0.8)).exp());
        let uptake = if glia { astro.buffer(k) } else { 0.0 };
        k += dt * (release * r - uptake - leak * k);
        k = k.max(0.0);
    }
    r
}

pub fn calcium_wave(coupled: bool) -> Vec<Vec<f64>> {
    let (n, steps) = (5usize, 140usize);
    let (dt, decay, drive): (f64, f64, f64) = (0.2, 0.08, 1.0);
    let couple = if coupled { 0.45 } else { 0.0 };
    let mut astros: Vec<Astrocyte> = (0..n).map(|_| Astrocyte::new(2.0, decay, couple, 0.07, dt)).collect();
    let mut trace = vec![vec![0.0; steps]; n];
    for t in 0..steps {
        let cas: Vec<f64> = astros.iter().map(|a| a.ca).collect();
        for i in 0..n {
            let inp = if i == 0 && t < 30 { drive } else { 0.0 };
            let neigh_sum = (if i > 0 { cas[i - 1] } else { 0.0 }) + (if i < n - 1 { cas[i + 1] } else { 0.0 });
            let n_neigh = (if i > 0 { 1.0 } else { 0.0 }) + (if i < n - 1 { 1.0 } else { 0.0 });
            astros[i].integrate(inp, neigh_sum - n_neigh * cas[i]);
            trace[i][t] = astros[i].ca;
        }
    }
    trace
}

pub fn tripartite_plasticity(glia_active: bool) -> f64 {
    let mut astro = Astrocyte::default_();
    let mut w = 0.0;
    for _ in 0..20 {
        astro.integrate(if glia_active { 1.0 } else { 0.0 }, 0.0);
        w += 0.2 * 1.0 * astro.d_serine();
    }
    w
}

pub fn metabolic_endurance(glia_supply: bool) -> f64 {
    let astro = Astrocyte::default_();
    let (mut energy, mut rates) = (1.0f64, vec![]);
    for _ in 0..50 {
        let r = energy.clamp(0.0, 1.0);
        let refuel = if glia_supply { astro.lactate(r) } else { 0.0 };
        energy = (energy - 0.06 * r + refuel).clamp(0.0, 1.0);
        rates.push(r);
    }
    rates[40..].iter().sum::<f64>() / 10.0
}

fn peak_time(series: &[f64]) -> usize {
    let mut bi = 0;
    let mut bv = f64::NEG_INFINITY;
    for (i, &x) in series.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astrocytes_compute() {
        // K⁺ buffering keeps firing stable; without it it runs away
        assert!(potassium_stability(true) < 0.6 && potassium_stability(false) > 0.9);

        // slow Ca²⁺ wave: distal astrocyte peaks later; without coupling it stays local
        let wave = calcium_wave(true);
        let no_wave = calcium_wave(false);
        let dmax = wave[3].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let nmax = no_wave[3].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(dmax > 0.1 && peak_time(&wave[3]) > peak_time(&wave[0]) + 5 && nmax < 0.02,
                "Ca wave should spread with delay only when coupled");

        // D-serine gates plasticity
        assert!(tripartite_plasticity(true) > 0.5 && tripartite_plasticity(false) < 0.05);

        // lactate fuels sustained firing
        assert!(metabolic_endurance(true) > 0.7 && metabolic_endurance(false) < 0.3);
    }
}
