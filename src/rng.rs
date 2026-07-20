//! rng.rs — a tiny deterministic PRNG, the replacement for torch.Generator/randn/rand/randint/multinomial.
//! SplitMix64 core + Box–Muller Gaussian. No external crate. Seeded → reproducible within Rust (it will
//! NOT reproduce torch's exact stream, so ported self-tests are validated by CONCLUSION, not bit-match).

pub struct Rng {
    s: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { s: seed.wrapping_add(0x9E3779B97F4A7C15) }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// uniform in [0, 1)
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// uniform in [lo, hi)
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// standard normal (mean 0, std 1) via Box–Muller
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// integer in [0, n)
    pub fn randint(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    /// a Vec of n standard-normal samples, each × scale
    pub fn randn_vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.normal() * scale).collect()
    }

    /// a Vec of n uniform[0,1) samples
    pub fn rand_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform()).collect()
    }

    /// sample one index from a (positive) weight vector, ∝ weight (torch.multinomial, num_samples=1)
    pub fn multinomial(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().map(|w| w.max(0.0)).sum();
        if total <= 0.0 {
            return self.randint(weights.len());
        }
        let mut r = self.uniform() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w.max(0.0);
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }

    /// in-place Fisher–Yates shuffle (torch.randperm applied to a slice)
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.randint(i + 1);
            v.swap(i, j);
        }
    }

    /// a random permutation of 0..n
    pub fn randperm(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        self.shuffle(&mut v);
        v
    }

    /// Poisson(lambda) via Knuth's algorithm (fine for the small λ these cells use)
    pub fn poisson(&mut self, lambda: f64) -> f64 {
        if lambda <= 0.0 {
            return 0.0;
        }
        let l = (-lambda).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= self.uniform();
            if p <= l {
                break;
            }
        }
        (k - 1) as f64
    }
}
