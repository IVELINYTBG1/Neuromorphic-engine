//! vec.rs — the tiny numeric helpers that replace the torch ops the cells actually use
//! (matvec, outer, softmax, topk, argmax, elementwise). Plain `Vec<f64>` / row-major matrices.
//! This is the whole "numeric library" — a few dozen lines, no crate.

/// y = x · W, where W is row-major (rows × cols), x has len rows → y has len cols.
pub fn matvec(x: &[f64], w: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut y = vec![0.0; cols];
    for i in 0..rows {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        let base = i * cols;
        for j in 0..cols {
            y[j] += xi * w[base + j];
        }
    }
    y
}

/// y = W · x, where W is row-major (rows × cols), x has len cols → y has len rows.
pub fn mv(w: &[f64], x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut y = vec![0.0; rows];
    for i in 0..rows {
        let base = i * cols;
        let mut acc = 0.0;
        for j in 0..cols {
            acc += w[base + j] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// W += a · outer(u, v)  (row-major, rows=len(u), cols=len(v))
pub fn add_outer(w: &mut [f64], u: &[f64], v: &[f64], a: f64) {
    let cols = v.len();
    for i in 0..u.len() {
        let ui = u[i] * a;
        if ui == 0.0 {
            continue;
        }
        let base = i * cols;
        for j in 0..cols {
            w[base + j] += ui * v[j];
        }
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn argmax(v: &[f64]) -> usize {
    let mut bi = 0;
    let mut bv = f64::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

pub fn max(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

pub fn min(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

pub fn sum(v: &[f64]) -> f64 {
    v.iter().sum()
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { sum(v) / v.len() as f64 }
}

/// indices of the k largest entries (descending). Small k → simple partial selection.
pub fn topk_indices(v: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

/// numerically-stable softmax over a temperature-scaled logit vector
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let m = max(logits);
    let ex: Vec<f64> = logits.iter().map(|z| (z - m).exp()).collect();
    let s: f64 = ex.iter().sum();
    ex.iter().map(|e| e / s).collect()
}

pub fn softmax_temp(logits: &[f64], temp: f64) -> Vec<f64> {
    let t = temp.max(1e-6);
    let scaled: Vec<f64> = logits.iter().map(|z| z / t).collect();
    softmax(&scaled)
}

#[inline]
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo { lo } else if x > hi { hi } else { x }
}

#[inline]
pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// L2 norm
pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Pearson correlation of two equal-length slices
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let ma = mean(a);
    let mb = mean(b);
    let ca: Vec<f64> = a.iter().map(|x| x - ma).collect();
    let cb: Vec<f64> = b.iter().map(|x| x - mb).collect();
    let denom = norm(&ca) * norm(&cb);
    if denom < 1e-12 { 0.0 } else { dot(&ca, &cb) / denom }
}
