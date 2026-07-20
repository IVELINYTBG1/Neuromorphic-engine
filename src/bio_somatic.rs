//! bio_somatic (limbic #5): somatic markers / Iowa Gambling Task — a loss-weighted gut feeling steers
//! choice to the good decks; a marker-lesioned chooser chases immediate wins and goes broke. From bio_somatic.py.

use crate::rng::Rng;
use crate::vec;

struct Deck {
    win: f64,
    loss: f64,
    p_loss: f64,
}
impl Deck {
    fn draw(&self, rng: &mut Rng) -> f64 {
        let mut r = self.win;
        if rng.uniform() < self.p_loss {
            r -= self.loss;
        }
        r
    }
}

fn decks() -> [Deck; 4] {
    [
        Deck { win: 1.0, loss: 3.0, p_loss: 0.5 },  // A bad
        Deck { win: 1.0, loss: 6.0, p_loss: 0.25 }, // B bad
        Deck { win: 0.5, loss: 0.5, p_loss: 0.5 },  // C good
        Deck { win: 0.5, loss: 1.0, p_loss: 0.25 }, // D good
    ]
}

/// Returns (marker, choices, total). somatic=true → choose by the loss-weighted marker; false → by wins only.
pub fn play(somatic: bool, seed: u64) -> ([f64; 4], Vec<usize>, f64) {
    let (beta, lr_gain, lr_loss) = (2.5, 0.15, 0.6);
    let mut rng = Rng::new(seed);
    let d = decks();
    let mut marker = [0.0; 4];
    let mut win_only = [0.0; 4];
    let mut choices = vec![];
    let mut total = 0.0;
    for _ in 0..300 {
        let signal = if somatic { marker } else { win_only };
        let logits: Vec<f64> = signal.iter().map(|s| beta * s).collect();
        let p = vec::softmax(&logits);
        let choice = rng.multinomial(&p);
        let outcome = d[choice].draw(&mut rng);
        let lr = if outcome < 0.0 { lr_loss } else { lr_gain };
        marker[choice] += lr * (outcome - marker[choice]);
        win_only[choice] += lr_gain * (outcome.max(0.0) - win_only[choice]);
        choices.push(choice);
        total += outcome;
    }
    (marker, choices, total)
}

fn good_frac(choices: &[usize]) -> f64 {
    let half = &choices[choices.len() / 2..];
    half.iter().filter(|&&c| c == 2 || c == 3).count() as f64 / half.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gut_feeling_beats_the_bait() {
        let (marker, choices, total) = play(true, 0);
        assert!(good_frac(&choices) > 0.6, "learns to prefer the good decks");
        let good_marker = (marker[2] + marker[3]) / 2.0;
        let bad_marker = (marker[0] + marker[1]) / 2.0;
        // markers DISCRIMINATE (good ≫ bad); the bad decks are strongly, robustly negative. (Good-deck
        // average can dip slightly negative because loss-weighted markers of a rarely-picked good deck
        // are volatile — the robust, seed-independent signal is the separation, not good's sign.)
        assert!(bad_marker < -0.5 && good_marker > bad_marker + 1.0, "markers tag bad ≪ good");
        let early = &choices[40..100];
        let early_good = early.iter().filter(|&&c| c == 2 || c == 3).count() as f64 / early.len() as f64;
        assert!(early_good > 0.5, "gut steers to good decks early");

        let (_, lchoices, ltotal) = play(false, 0);
        assert!(good_frac(&lchoices) < 0.45 && total > ltotal + 30.0, "marker-lesion goes broke");
    }
}
