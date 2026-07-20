//! bio_lang (learning #5, synthesis): a char language model from the substrate — delay-line context +
//! conjunctive group cells (bio_conj) + sampling generation. Next-char accuracy rises with context depth;
//! temp-sampling generates coherent, real-word text. Local learning, no backprop. Ported from bio_lang.py.

use crate::bio_conj::PolyGroupReadout;
use crate::rng::Rng;
use crate::vec;
use std::collections::HashSet;

pub fn corpus() -> String {
    let base = "alpha thinks slowly and beta feels deeply. alpha is patient but beta is restless. \
                they share one heart and the heart is phill. ";
    base.repeat(12)
}

pub fn encode(text: &str) -> (Vec<usize>, Vec<char>) {
    let mut vocab: Vec<char> = text.chars().collect();
    vocab.sort();
    vocab.dedup();
    let seq: Vec<usize> = text.chars().map(|c| vocab.iter().position(|&v| v == c).unwrap()).collect();
    (seq, vocab)
}

/// delay-line context pairs: (flattened k×m ring, next symbol)
pub fn make_pairs(seq: &[usize], k: usize, m: usize) -> Vec<(Vec<f64>, usize)> {
    let mut ring = vec![0.0; k * m];
    let mut pairs = vec![];
    for t in 0..seq.len() - 1 {
        for i in (1..k).rev() {
            for j in 0..m {
                ring[i * m + j] = ring[(i - 1) * m + j];
            }
        }
        for j in 0..m {
            ring[j] = 0.0;
        }
        ring[seq[t]] = 1.0;
        pairs.push((ring.clone(), seq[t + 1]));
    }
    pairs
}

pub fn teacher_forced_acc(net: &PolyGroupReadout, pairs: &[(Vec<f64>, usize)]) -> f64 {
    pairs.iter().filter(|(x, y)| net.predict(x) == *y).count() as f64 / pairs.len() as f64
}

pub fn generate(net: &PolyGroupReadout, seed: &[usize], n: usize, k: usize, m: usize, temp: f64, rng: &mut Rng) -> Vec<usize> {
    let mut ring = vec![0.0; k * m];
    let mut push = |ring: &mut Vec<f64>, s: usize| {
        for i in (1..k).rev() {
            for j in 0..m {
                ring[i * m + j] = ring[(i - 1) * m + j];
            }
        }
        for j in 0..m {
            ring[j] = 0.0;
        }
        ring[s] = 1.0;
    };
    for &s in seed {
        push(&mut ring, s);
    }
    let mut out = seed.to_vec();
    for _ in 0..n {
        let p = vec::softmax_temp(&net.logits(&ring), temp);
        let nxt = rng.multinomial(&p);
        out.push(nxt);
        push(&mut ring, nxt);
    }
    out
}

pub fn real_word_fraction(text: &str, vocab_words: &HashSet<String>) -> f64 {
    let toks: Vec<String> = text.split(' ').map(|t| t.trim_matches('.').to_string()).filter(|t| !t.is_empty()).collect();
    if toks.is_empty() {
        return 0.0;
    }
    toks.iter().filter(|t| vocab_words.contains(*t)).count() as f64 / toks.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_language_model_from_the_substrate() {
        let text = corpus();
        let (seq, vocab) = encode(&text);
        let m = vocab.len();
        let words: HashSet<String> = text.split(' ').map(|w| w.trim_matches('.').to_string()).filter(|w| !w.is_empty()).collect();

        // next-char accuracy is high with context depth k=4 (conjunctive group readout)
        let k = 4;
        let pairs = make_pairs(&seq, k, m);
        let mut net = PolyGroupReadout::new(m * k, m, 400, 24, 0);
        net.train(&pairs, 20);
        assert!(teacher_forced_acc(&net, &pairs) >= 0.85, "char-LM accuracy high");

        // temp-sampling generates coherent, mostly-real-word text
        let seed: Vec<usize> = "alpha ".chars().map(|c| vocab.iter().position(|&v| v == c).unwrap()).collect();
        let mut g = Rng::new(0);
        let gen = generate(&net, &seed, 120, k, m, 0.4, &mut g);
        let samp: String = gen.iter().map(|&i| vocab[i]).collect();
        assert!(real_word_fraction(&samp, &words) >= 0.7, "generates real words: {:?}", samp);
    }
}
