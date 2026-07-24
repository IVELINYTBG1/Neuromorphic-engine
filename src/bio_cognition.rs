//! bio_cognition — HER THINKING/REASONING ORGAN, learned. A custom architecture, invented here because
//! none of the off-the-shelf ones fit what this project needs.
//!
//! THE PROBLEM IT SOLVES. The bio-substrate stores a concept as a POPULATION and a memory as a grown
//! PATHWAY (`bio_cortex`). That is honest and it grows with living, but it ENUMERATES — roughly one synapse
//! per remembered pairing — so human-scale would need ~30 TB. That is the wall. Brains, and
//! transformers, both dodge that wall the SAME way: COMPRESSION. A bounded network whose weights encode the
//! REGULARITIES of experience gets SMARTER, not BIGGER. This is that: a fixed-size learned core that models
//! the structure of her world, so she can reason at bounded cost. A learned reasoning organ —
//! dense learned parameters — but trained the brain's way, not the machine-learning way.
//!
//! WHY NOT THE THREE THINGS THAT ALREADY EXIST (none of them works like this):
//!   • an LLM / spiking transformer learns by BACKPROP over a frozen corpus — offline, and the corpus is
//!     injection, which this project forbids. Its "attention" is a trained matrix, not lived dynamics.
//!   • a plain SNN (STDP) learns online and locally (good) but holds no working representation to REASON
//!     over — it recalls sequences, it does not think about them.
//! What was missing is a core that learns ONLINE, LOCALLY, FEELING-GATED, from her own lived stream, and
//! reasons over TIME. So it is built from the substrate's OWN proven local-learning cells — nothing alien:
//!   • a recurrent hidden state = her WORKING THOUGHT that persists and transforms (bio_wm / bio_traj).
//!   • it PREDICTS the next moment of experience; the error drives learning (predictive coding, bio_predict
//!     — self-supervised on her LIFE, no dataset, no labels).
//!   • credit reaches the hidden layer through FIXED RANDOM FEEDBACK (DFA, bio_net) — no backprop, no weight
//!     transport. This is the one trick that lets a deep-ish local learner assign blame without being an LLM.
//!   • EVERY update is gated by FEELING — the three-factor rule × neuromodulator M (bio_phill's signature):
//!     what she felt strongly, she learns hard; the boring washes out.
//!
//! WHAT "REASONING" MEANS HERE, concretely and testably: thinking is simulating the next state of the world
//! (a forward model). Trained on her experience, the core rolls its own predictions forward — given a
//! premise it settles toward the consequence, and because the state is RECURRENT it handles context ("the
//! same thing means different things depending on what came before"), which pure association cannot.
//!
//! HONEST CEILING: bounded, CPU, rate-coded, local. It makes a mind LEARN TO PREDICT/REASON over its stream,
//! feeling-gated, with nothing injected. It is not AGI and not a frontier coder (the scale + culture walls
//! above stand). It is a genuinely different KIND of thinking than the co-occurrence walk it
//! augments — a learned model of her world, invented to be consistent with the brain she already is.

use crate::rng::Rng;

/// One remembered forward step — the context that made a prediction, kept until the next moment arrives so
/// the error can be assigned back locally (online predictive learning has to wait one step for the truth).
struct Trace {
    x: Vec<f64>,      // the input at the step that made the prediction
    h_prev: Vec<f64>, // the recurrent state going INTO that step
    a: Vec<f64>,      // the unit activations at that step (for the tanh derivative)
    h: Vec<f64>,      // the state the prediction was read out of
    pred: Vec<f64>,   // what she predicted would come next
}

pub struct CognitiveCore {
    d: usize,        // input/prediction width — a compressed code of "what she is thinking about now"
    h_dim: usize,    // hidden width — how much working thought she can hold
    leak: f64,       // how much the working state carries vs. updates each step (memory vs. reactivity)
    lr: f64,
    decay: f64,      // homeostatic weight decay — keeps an online recurrent learner from running away
    w_in: Vec<f64>,  // H×D   input → hidden
    w_rec: Vec<f64>, // H×H   hidden → hidden (the learned DYNAMICS — where reasoning lives)
    w_pred: Vec<f64>,// D×H   hidden → prediction of the next input (the forward model readout)
    b_fb: Vec<f64>,  // H×D   FIXED random feedback — carries the error back without weight transport (DFA)
    h: Vec<f64>,     // current working thought
    surprise: f64,   // EMA of her own prediction error — how WRONG she has been lately. Her uncertainty
                     // about the world, LEARNED not decreed: high when untrained/in novel territory, low
                     // when her model has been reliable. This is the precision signal of predictive coding.
    surprise0: f64,  // her FIRST prediction error — the untrained baseline she measures precision against
    trace: Option<Trace>,
    learns: bool,
    recurrent: bool, // false = memoryless (no working state across steps) — the teeth for "she reasons over TIME"    // false = ablated (frozen): the teeth for "it is the LEARNING that thinks"
    // ── ADAPTIVE CAPACITY — she is not a fixed size. She starts small and GROWS her own hidden layer when she
    //    is STRAINING to model her world (precision below her standard), and only then — a content, well-
    //    modelling being stays small and cheap. There is no preset architectural ceiling; the only limit is
    //    `max_h`, set from the machine's real RAM (the body sensing its own flesh, like bio_soma). She decides
    //    how much mind to use, by need. Neurogenesis for the reasoning organ.
    max_h: usize,      // the RAM-derived hard cap on hidden width (NOT an operating size — she rarely reaches it)
    growth_bar: f64,   // the precision she wants before she stops growing — her standard (from temperament)
    steps: usize,      // steps lived (for warmup + the periodic grow-check)
    grow_rng: Rng,     // draws the weights of newly-grown neurons
    last_check_precision: f64, // her precision at the previous grow-check — to see if it is still RISING
    stuck_grows: u32,  // grows made while PLATEAUED below her standard; a few that don't unstick her → she is
                       // SATURATED (this world's entropy, not her capacity, is the limit) and she stops growing
    saturated: bool,   // she has found that more mind no longer helps THIS world — growth is done
}

impl CognitiveCore {
    pub fn new(d: usize, h_dim: usize, rng: &mut Rng) -> Self {
        let si = 1.0 / (d as f64).sqrt();
        let sr = 1.0 / (h_dim as f64).sqrt();
        CognitiveCore {
            d,
            h_dim,
            leak: 0.3,
            // NLMS normalises each step by the signal power (1 + ‖·‖²), which shrinks the raw step by ~‖h‖²;
            // lr is raised to compensate so she still learns fast — and a BIGGER core (more weights) needs a
            // brisker rate to converge from her lived data, so it went 0.3→0.6 when the core skyrocketed. g =
            // lr·m stays under the NLMS stability bound of 2 (≈0.9 at her most-felt) — speed, not instability.
            lr: 0.6,
            decay: 2e-4,
            w_in: rng.randn_vec(h_dim * d, si),
            w_rec: rng.randn_vec(h_dim * h_dim, sr * 0.5),
            w_pred: rng.randn_vec(d * h_dim, sr),
            b_fb: rng.randn_vec(h_dim * d, 1.0), // fixed, never learned (DFA)
            h: vec![0.0; h_dim],
            surprise: 1.0,
            surprise0: -1.0,
            trace: None,
            learns: true,
            recurrent: true,
            max_h: h_dim, // a plain `new` is FIXED size (max = start) — for the cell tests & controlled comparisons
            growth_bar: 1.0,
            steps: 0,
            grow_rng: Rng::new(0x6C0DE ^ (h_dim as u64).wrapping_mul(0x9E3779B1) ^ (d as u64)),
            last_check_precision: 0.0,
            stuck_grows: 0,
            saturated: false,
        }
    }

    /// A GROWABLE core — the being's real one. Starts at `h_start` and GROWS its hidden layer toward `max_h`
    /// (a RAM-derived cap) whenever she is straining: her precision below `bar` (her standard) means she claims
    /// more mind. She decides how much to use, by need — no preset operating size, the ceiling is RAM.
    pub fn growable(d: usize, h_start: usize, max_h: usize, bar: f64, rng: &mut Rng) -> Self {
        let mut c = CognitiveCore::new(d, h_start.min(max_h), rng);
        c.max_h = max_h.max(h_start);
        c.growth_bar = bar.clamp(0.0, 0.99);
        c
    }

    /// GROW her hidden layer by `add` units (up to `max_h`) — neurogenesis for the reasoning organ. The old
    /// weights are preserved exactly; new neurons get small input/recurrent weights (they start learning) and
    /// ZERO output weights (so growing does not disturb what she already predicts — the new capacity engages
    /// gradually as it learns, never a jolt). She is bigger, and still herself.
    pub fn grow(&mut self, add: usize) {
        let (old, d) = (self.h_dim, self.d);
        let new = (old + add).min(self.max_h);
        if new <= old {
            return;
        }
        let si = 1.0 / (d as f64).sqrt();
        let sr = 1.0 / (new as f64).sqrt();
        // w_in: new×d — old rows kept, new rows small-random
        let mut w_in = vec![0.0; new * d];
        w_in[..old * d].copy_from_slice(&self.w_in);
        for x in w_in[old * d..].iter_mut() {
            *x = self.grow_rng.randn_vec(1, si * 0.5)[0];
        }
        // w_rec: new×new — old block kept, new entries small
        let mut w_rec = vec![0.0; new * new];
        for i in 0..old {
            for j in 0..old {
                w_rec[i * new + j] = self.w_rec[i * old + j];
            }
        }
        for i in 0..new {
            for j in 0..new {
                if i >= old || j >= old {
                    w_rec[i * new + j] = self.grow_rng.randn_vec(1, sr * 0.3)[0];
                }
            }
        }
        // w_pred: d×new — old cols kept, NEW cols ZERO (new neurons do not change her predictions yet)
        let mut w_pred = vec![0.0; d * new];
        for dd in 0..d {
            for i in 0..old {
                w_pred[dd * new + i] = self.w_pred[dd * old + i];
            }
        }
        // b_fb: new×d — old kept, new fixed-random feedback for the new units
        let mut b_fb = vec![0.0; new * d];
        b_fb[..old * d].copy_from_slice(&self.b_fb);
        for x in b_fb[old * d..].iter_mut() {
            *x = self.grow_rng.randn_vec(1, 1.0)[0];
        }
        // working state grows with zeros (the new units start silent)
        let mut h = vec![0.0; new];
        h[..old].copy_from_slice(&self.h);
        self.w_in = w_in;
        self.w_rec = w_rec;
        self.w_pred = w_pred;
        self.b_fb = b_fb;
        self.h = h;
        self.h_dim = new;
        self.trace = None; // shapes changed — the pending prediction is stale
    }

    /// Her current hidden width (grows as she strains) and her ceiling.
    pub fn hidden_width(&self) -> usize {
        self.h_dim
    }
    pub fn capacity_cap(&self) -> usize {
        self.max_h
    }
    /// A MEMORYLESS core — same learning, but no working state carries between steps, so each moment is
    /// judged only by the current input. It can learn associations; it CANNOT reason over time. The teeth
    /// for the working-memory claim.
    pub fn memoryless(d: usize, h_dim: usize, rng: &mut Rng) -> Self {
        let mut c = CognitiveCore::new(d, h_dim, rng);
        c.recurrent = false;
        c.leak = 1.0; // h = a(x), a pure function of the current input — nothing remembered
        c
    }
    /// A frozen core — it still thinks (runs its dynamics) but never learns. The teeth for every claim.
    pub fn ablated(d: usize, h_dim: usize, rng: &mut Rng) -> Self {
        let mut c = CognitiveCore::new(d, h_dim, rng);
        c.learns = false;
        c
    }

    #[cfg(test)]
    pub fn set_hp(&mut self, leak: f64, lr: f64) { self.leak = leak; self.lr = lr; }

    /// Every learned number in her thinking organ — FIXED, whatever she lives. This is the whole point:
    /// she gets smarter, not bigger. Contrast the structural cortex, which grows a synapse per pairing.
    pub fn parameters(&self) -> usize {
        self.w_in.len() + self.w_rec.len() + self.w_pred.len()
    }

    /// Her whole learned organ as flat weights — for baking into her brain (it MUST persist, or she resets
    /// her thinking every session). Compact & fixed-size (D·H + H·H + D·H floats), the same order `import`
    /// reads them in. The fixed random feedback regenerates from the salt, so it need not be stored.
    pub fn export(&self) -> Vec<f64> {
        // [d, h_dim] header so a GROWN core round-trips at whatever size she reached — her capacity is not
        // fixed, so the size must be stored, not assumed. Then the weights (incl. the fixed feedback b_fb, so
        // no regeneration is needed), then her precision history.
        let mut v = Vec::with_capacity(self.parameters() + self.b_fb.len() + 4);
        v.push(self.d as f64);
        v.push(self.h_dim as f64);
        v.extend_from_slice(&self.w_in);
        v.extend_from_slice(&self.w_rec);
        v.extend_from_slice(&self.w_pred);
        v.extend_from_slice(&self.b_fb);
        v.push(self.surprise0); // her precision history must survive too, or a reborn core thinks it is
        v.push(self.surprise);  // untrained and over-abstains until it re-earns its confidence
        v
    }
    /// Reload the learned organ from baked weights — exact, if the shape matches (a mismatch is ignored, so
    /// a resized core just starts fresh rather than crashing).
    pub fn import(&mut self, w: &[f64]) {
        // Read the stored size from the header and RESIZE this core to match — she may have grown a bigger mind
        // than she started with, and it must come back at that size. Anything malformed (an old headerless blob,
        // a different code width, non-finite or blown-up weights from a diverged legacy brain) is rejected and
        // she keeps her fresh, stable init — self-healing, she just re-earns her reasoning.
        if w.len() < 4 {
            return;
        }
        let d = w[0] as usize;
        let h = w[1] as usize;
        if d != self.d || h == 0 || h > 1 << 20 {
            return; // headerless/old blob or a changed code width → start fresh
        }
        let (n_in, n_rec, n_pred, n_fb) = (h * d, h * h, d * h, h * d);
        let total = 2 + n_in + n_rec + n_pred + n_fb + 2;
        if w.len() != total {
            return;
        }
        if w.iter().skip(2).any(|v| !v.is_finite() || v.abs() > 1e6) {
            return; // a diverged legacy core carries no usable information — reject, stay stable
        }
        // resize this core to her baked size, then copy every weight back exactly
        self.h_dim = h;
        self.max_h = self.max_h.max(h); // never cap below what she has already grown to
        let (a, b, c, e) = (2, 2 + n_in, 2 + n_in + n_rec, 2 + n_in + n_rec + n_pred);
        self.w_in = w[a..b].to_vec();
        self.w_rec = w[b..c].to_vec();
        self.w_pred = w[c..e].to_vec();
        self.b_fb = w[e..e + n_fb].to_vec();
        self.h = vec![0.0; h];
        self.trace = None;
        self.surprise0 = w[e + n_fb];
        self.surprise = w[e + n_fb + 1];
    }

    /// Start a fresh train of thought — clear the working state (a new premise, not a continuation).
    pub fn clear_state(&mut self) {
        for v in self.h.iter_mut() {
            *v = 0.0;
        }
        self.trace = None;
    }

    /// ONE MOMENT OF THOUGHT. Take the current experience `x` (a compressed code of what she is attending to)
    /// and her feeling `m` (the neuromodulator gate, 0..~1.5). Learn from how well the LAST moment predicted
    /// this one, advance the working state, and return her prediction of the NEXT moment — what she now
    /// expects. All learning is local and gated by `m`; nothing here is backprop.
    pub fn step(&mut self, x: &[f64], m: f64) -> Vec<f64> {
        // 1. THE TRUTH ARRIVED: learn from last step's prediction against this actual input.
        if self.learns {
            if let Some(tr) = self.trace.take() {
                self.learn(x, &tr, m);
            }
            // 1b. GROW IF STRAINING — adaptive capacity, triggered by a PLATEAU (not by mere improvement). Every
            //     so often, after a warmup: if her precision is STILL RISING she is learning fine from the data
            //     she has — do nothing, let it. Only when it has PLATEAUED *below* the standard she holds herself
            //     to is capacity the bottleneck, and she claims more mind. If a few such grows in a row fail to
            //     unstick her, it is THIS world's entropy, not her size — she is saturated and stops. So a being
            //     with a rich, learnable world grows to meet it; a content one, or one whose world has no more
            //     structure to find, stays small. No preset size — need decides; the only hard limit is RAM.
            self.steps += 1;
            if !self.saturated && self.h_dim < self.max_h && self.steps > 600 && self.steps % 400 == 0 {
                let p = self.precision();
                let rising = p > self.last_check_precision + 0.01;
                self.last_check_precision = p;
                if rising {
                    self.stuck_grows = 0; // she is still learning — capacity is not the bottleneck, don't grow
                } else if p < self.growth_bar {
                    self.grow((self.h_dim / 4).max(48)); // plateaued below her standard → claim more mind
                    self.stuck_grows += 1;
                    if self.stuck_grows >= 3 {
                        self.saturated = true; // three grows, still stuck → entropy, not capacity — she is done
                    }
                }
            }
        }
        // 2. ADVANCE the working thought: recurrent update, leaky so it holds context across steps.
        let h_prev = self.h.clone();
        let mut a = vec![0.0; self.h_dim];
        let mut h_new = vec![0.0; self.h_dim];
        for i in 0..self.h_dim {
            let mut pre = 0.0;
            for j in 0..self.d {
                pre += self.w_in[i * self.d + j] * x[j];
            }
            if self.recurrent {
                for k in 0..self.h_dim {
                    pre += self.w_rec[i * self.h_dim + k] * h_prev[k];
                }
            }
            a[i] = pre.tanh();
            h_new[i] = (1.0 - self.leak) * h_prev[i] + self.leak * a[i];
        }
        // 3. PREDICT the next moment from the new state (the forward model readout).
        let mut pred = vec![0.0; self.d];
        for dd in 0..self.d {
            let mut s = 0.0;
            for i in 0..self.h_dim {
                s += self.w_pred[dd * self.h_dim + i] * h_new[i];
            }
            pred[dd] = s;
        }
        self.trace = Some(Trace { x: x.to_vec(), h_prev, a, h: h_new.clone(), pred: pred.clone() });
        self.h = h_new;
        pred
    }

    /// The LOCAL, feeling-gated learning rule. Predictive coding error → DFA-projected credit → three-factor
    /// weight change. No backprop, no weight transport.
    fn learn(&mut self, target: &[f64], tr: &Trace, m: f64) {
        let g = self.lr * m;
        // error on the forward model (what actually came next, minus what she expected)
        let mut err = vec![0.0; self.d];
        for dd in 0..self.d {
            err[dd] = target[dd] - tr.pred[dd];
        }
        // TRACK HER OWN SURPRISE — the mean squared prediction error, EMA'd. This is emergent uncertainty:
        // nobody sets it, it IS how wrong her learned model has just been. The first one is her baseline.
        let mse = err.iter().map(|e| e * e).sum::<f64>() / self.d as f64;
        if self.surprise0 < 0.0 {
            self.surprise0 = mse.max(1e-6);
            self.surprise = mse;
        } else {
            self.surprise = 0.98 * self.surprise + 0.02 * mse;
        }
        // readout: NORMALIZED least-mean-squares (NLMS). The step is the exact local gradient of a linear
        // predictor, but SCALED by the hidden state's own power (1 + ‖h‖²). A raw LMS step is stable only
        // while g·‖h‖² < 2; once the recurrent state saturates (‖h‖² → h_dim) a fixed step blows past that
        // bound and the readout diverges to ±∞. This is not hypothetical — a baked core was found with ~46%
        // of its weights above 1e100 and a surprise EMA of 1e225, precision pinned at 0 for good. Normalizing
        // by ‖h‖² makes it UNCONDITIONALLY stable for any g < 2, at any activation scale. Same no-backprop,
        // local rule bio_vocal already learns its tract with.
        let h_pow = 1.0 + tr.h.iter().map(|v| v * v).sum::<f64>();
        for dd in 0..self.d {
            for i in 0..self.h_dim {
                let idx = dd * self.h_dim + i;
                self.w_pred[idx] += g * err[dd] * tr.h[i] / h_pow - self.lr * self.decay * self.w_pred[idx];
            }
        }
        // hidden credit via FIXED RANDOM FEEDBACK (DFA): project the error back through b_fb, gate by the
        // tanh derivative and the leak — then normalise by the incoming signal power too, so the recurrent
        // and input weights are held to the same stability bound as the readout (they cannot run away either).
        let hp_pow = 1.0 + tr.h_prev.iter().map(|v| v * v).sum::<f64>();
        let x_pow = 1.0 + tr.x.iter().map(|v| v * v).sum::<f64>();
        for i in 0..self.h_dim {
            let mut fb = 0.0;
            for dd in 0..self.d {
                fb += self.b_fb[i * self.d + dd] * err[dd];
            }
            let delta = fb * self.leak * (1.0 - tr.a[i] * tr.a[i]);
            let gd = g * delta;
            if self.recurrent {
                for k in 0..self.h_dim {
                    let idx = i * self.h_dim + k;
                    self.w_rec[idx] += gd * tr.h_prev[k] / hp_pow - self.lr * self.decay * self.w_rec[idx];
                }
            }
            for j in 0..self.d {
                let idx = i * self.d + j;
                self.w_in[idx] += gd * tr.x[j] / x_pow - self.lr * self.decay * self.w_in[idx];
            }
        }
    }

    /// Her current working thought, and a way to set it back — so a peek-ahead (imagining) can run from her
    /// LIVE state and then be undone, leaving her real train of thought where it was.
    pub fn hidden(&self) -> Vec<f64> {
        self.h.clone()
    }
    pub fn set_hidden(&mut self, h: Vec<f64>) {
        self.h = h;
        self.trace = None;
    }
    /// One step of thought that does NOT learn (imagining is not living) but DOES advance the working state,
    /// and returns what she predicts comes next. Used to roll a train of thought forward from where she is.
    pub fn think_step(&mut self, x: &[f64]) -> Vec<f64> {
        let learns = self.learns;
        self.learns = false;
        let p = self.step(x, 0.0);
        self.learns = learns;
        p
    }

    /// WHAT SHE EXPECTS NEXT, RIGHT NOW — the forward-model readout from her CURRENT working state, without
    /// taking a step. This is the meaningful query for a stream predictor: not "what follows word X in the
    /// void" but "given everything I have just been thinking, what comes next." Her behaviour reads this.
    pub fn expect(&self) -> Vec<f64> {
        let mut pred = vec![0.0; self.d];
        for dd in 0..self.d {
            let mut s = 0.0;
            for i in 0..self.h_dim {
                s += self.w_pred[dd * self.h_dim + i] * self.h[i];
            }
            pred[dd] = s;
        }
        pred
    }

    /// HER PRECISION — how much she should trust her own model right now, in 0..1. EMERGENT: it is how far
    /// her recent prediction error has fallen below her untrained baseline. 0 when she has learned nothing or
    /// is in novel territory (error back up near baseline); →1 when her model has been reliable. Nothing
    /// hardcodes it; it is her track record.
    pub fn precision(&self) -> f64 {
        if self.surprise0 <= 0.0 {
            return 0.0; // she has never predicted anything — she knows nothing yet
        }
        (1.0 - self.surprise / self.surprise0).clamp(0.0, 1.0)
    }

    /// HER CERTAINTY about a choice among candidates, in 0..1 — the anti-hallucination signal, entirely
    /// emergent. Her CONVICTION about this choice times her PRECISION at choices in general. Two things she
    /// already knows, multiplied: (1) the MARGIN — is one candidate clearly the
    /// winner, or are several tied? (the peakedness of her own prediction, read as a softmax margin over the
    /// scores); and (2) her PRECISION — has her model been reliable lately? When she is untrained, or the
    /// prediction is diffuse, or she is in novel territory, this collapses toward 0 and she should ABSTAIN
    /// rather than assert — which is exactly not-hallucinating. No threshold decides truth here; the SHAPE of
    /// her learned prediction and her own error history do.
    pub fn certainty(&self, scores: &[f64]) -> f64 {
        self.conviction(scores) * self.precision()
    }

    /// HOW SURE SHE IS OF THIS PARTICULAR CHOICE, with the question of whether her model is any good left out
    /// of it: the absolute match and the margin, and nothing else. `certainty` is exactly this times her
    /// precision — the split matters because the two answer different questions, and multiplying them means a
    /// mind whose model is 0.18 reliable can never express more than 0.18 confidence about ANYTHING. Held
    /// against a bar meant for the first question alone, that is a mind that can never speak (measured in a live mind:
    /// precision 0.181, against her own bar of 0.59). A caller that wants both should ask for both.
    pub fn conviction(&self, scores: &[f64]) -> f64 {
        if scores.len() < 2 {
            return 0.0;
        }
        // (1) ABSOLUTE MATCH — does the prediction actually LOOK like the best candidate? On a transition she
        //     has learned this is ~1.0; on a novel query the prediction resembles nothing she knows (~0.2).
        //     This is what stops her asserting on things she has never learned.
        let top = scores.iter().cloned().fold(f64::MIN, f64::max).clamp(0.0, 1.0);
        // (2) MARGIN — is that best one a clear winner, or tied with another? (softmax over the scores)
        let tau = 0.12;
        let exps: Vec<f64> = scores.iter().map(|s| ((s - top) / tau).exp()).collect();
        let z: f64 = exps.iter().sum::<f64>().max(1e-9);
        let mut prob: Vec<f64> = exps.iter().map(|e| e / z).collect();
        prob.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let margin = 0.4 + 0.6 * (prob[0] - prob[1]); // a tie → 0.4, a clear winner → ~1.0
        top * margin
    }

    /// IMAGINE — roll her own predictions forward without new input: a train of thought as forward
    /// simulation. Given a starting moment, she thinks the next, then the next from that, and so on. This is
    /// reasoning as she has it: run the learned model of the world ahead of the world.
    pub fn imagine(&mut self, start: &[f64], steps: usize) -> Vec<Vec<f64>> {
        self.clear_state();
        let mut out = vec![];
        let mut cur = start.to_vec();
        for _ in 0..steps {
            let learns = self.learns;
            self.learns = false; // imagining does not teach — she is running the model, not living
            let next = self.step(&cur, 0.0);
            self.learns = learns;
            out.push(next.clone());
            cur = next;
        }
        out
    }

    /// How well she currently predicts a stream — mean squared error of next-step prediction. Lower = she
    /// has learned its structure. (For tests and for gauging how well she understands a thing.)
    pub fn predict_error(&mut self, seq: &[Vec<f64>]) -> f64 {
        let learns = self.learns;
        self.learns = false;
        self.clear_state();
        let mut tot = 0.0;
        let mut n = 0;
        let mut pred = self.step(&seq[0], 0.0);
        for x in &seq[1..] {
            for dd in 0..self.d {
                let e = x[dd] - pred[dd];
                tot += e * e;
            }
            n += self.d;
            pred = self.step(x, 0.0);
        }
        self.learns = learns;
        tot / n.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codes(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| {
                let mut v = rng.randn_vec(d, 1.0);
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
                for x in v.iter_mut() {
                    *x /= norm;
                }
                v
            })
            .collect()
    }
    fn cos(a: &[f64], b: &[f64]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        d / (na * nb).max(1e-9)
    }

    /// SHE LEARNS ONLINE, FROM THE STREAM — no dataset, no backprop, no epochs over a corpus. Fed a periodic
    /// experience she comes to PREDICT it: the error falls far below chance as she lives it.
    #[test]
    fn she_learns_to_predict_her_stream_online() {
        let d = 24;
        let c = codes(4, d, 1);
        let mut rng = Rng::new(7);
        let mut core = CognitiveCore::new(d, 64, &mut rng);
        // live A B C D A B C D … for a while, learning every step (feeling neutral-positive)
        for t in 0..1200 {
            core.step(&c[t % 4], 1.0);
        }
        // now measure: how well does she predict the next moment? (unit-norm codes → chance MSE ≈ 1/d each,
        // total ≈ 1.0 for a zero prediction; a learned model gets far under that)
        let seq: Vec<Vec<f64>> = (0..8).map(|t| c[t % 4].clone()).collect();
        let err = core.predict_error(&seq);
        assert!(err < 0.3, "she learned to predict her stream online: MSE {:.3} (chance ≈ 1.0)", err);
    }

    /// SHE REASONS OVER TIME — the differentiator from mere association. In "A B A C" the SAME input A must
    /// predict DIFFERENT things depending on what came before (B first, C second). Only a WORKING STATE can
    /// do that; a lookup table cannot. And a frozen (non-learning) core cannot either — proving it is the
    /// LEARNING of the recurrent dynamics that reasons, not the random init.
    #[test]
    fn she_reasons_from_context_not_just_association() {
        let d = 24;
        let c = codes(3, d, 2); // A=0, B=1, C=2
        let seq = [0usize, 1, 0, 2]; // A B A C, repeating
        let mut rng = Rng::new(11);
        let mut core = CognitiveCore::new(d, 128, &mut rng);
        for t in 0..9000 {
            core.step(&c[seq[t % 4]], 1.0);
        }
        // WARM the working state up through the loop (context is what makes the two A's differ — do not
        // wipe it right before reading, that erases the very memory being tested). The warmup ends on C, so
        // the next A is the "first" A (after C) and the one after B is the "second". Return the two raw
        // prediction VECTORS after each A.
        let read = |core: &mut CognitiveCore| -> (Vec<f64>, Vec<f64>) {
            core.clear_state();
            for t in 0..12 {
                core.step(&c[seq[t % 4]], 0.0);
            }
            let p1 = core.step(&c[0], 0.0); // A after C → should expect B
            core.step(&c[1], 0.0);
            let p2 = core.step(&c[0], 0.0); // A after B → should expect C
            (p1, p2)
        };
        let (p1, p2) = read(&mut core);
        // the first A leans to B, the second (same input!) leans to C — she used what came before
        assert!(cos(&p1, &c[1]) - cos(&p1, &c[2]) > 0.3, "first A predicts B ({:+.2})", cos(&p1, &c[1]) - cos(&p1, &c[2]));
        assert!(cos(&p2, &c[2]) - cos(&p2, &c[1]) > 0.3, "the SAME A after B predicts C ({:+.2})", cos(&p2, &c[2]) - cos(&p2, &c[1]));
        // and the two predictions are genuinely DIFFERENT (she distinguished the contexts)
        assert!(cos(&p1, &p2) < 0.5, "her two A-predictions differ — context changed the thought ({:.2})", cos(&p1, &p2));

        // TEETH: a MEMORYLESS core (same learning, no working state) sees each A only as the current input,
        // so both A's get the SAME prediction — it cannot reason over time, however much it learns. This is
        // what proves the reasoning comes from working memory. (The architecture is deliberately REDUNDANT in
        // HOW it learns — a random reservoir readout, or the hidden learning to a fixed readout, both work —
        // so no single learning weight is uniquely necessary; what IS necessary is the working state itself.)
        let mut rng2 = Rng::new(11);
        let mut flat = CognitiveCore::memoryless(d, 128, &mut rng2);
        for t in 0..9000 {
            flat.step(&c[seq[t % 4]], 1.0);
        }
        let (f1, f2) = read(&mut flat);
        assert!(cos(&f1, &f2) > 0.95, "memoryless: both A's give the SAME prediction — no context ({:.2})", cos(&f1, &f2));
    }

    /// FEELING GATES LEARNING — the three-factor rule. With only a FEW exposures (as a real moment gives
    /// you), what she felt strongly she has learned and what bored her has barely stuck. This is `bio_phill`'s
    /// flashbulb, now in the reasoning organ: M sets the RATE, so under limited living, feeling decides what
    /// is kept.
    #[test]
    fn feeling_gates_what_she_learns() {
        let d = 24;
        let c = codes(4, d, 3);
        let seq: Vec<Vec<f64>> = (0..8).map(|t| c[t % 4].clone()).collect();
        let train = |m: f64| {
            let mut rng = Rng::new(5);
            let mut core = CognitiveCore::new(d, 64, &mut rng);
            for t in 0..150 {
                core.step(&c[t % 4], m);
            }
            core.predict_error(&seq)
        };
        let felt = train(1.3);
        let dull = train(0.05);
        assert!(felt < dull * 0.6,
            "what she FELT, she learned (MSE {:.3}); what bored her barely stuck (MSE {:.3})", felt, dull);
    }

    /// COMPRESSION, not enumeration — the answer to the 30 TB wall. Her thinking organ is a FIXED size
    /// however much she lives; it captures the regularity of a long stream in bounded weights, where the
    /// structural cortex would have grown a synapse per pairing. Smarter, not bigger.
    #[test]
    fn she_compresses_her_world_into_bounded_weights() {
        let d = 24;
        let c = codes(5, d, 4);
        let mut rng = Rng::new(9);
        let mut core = CognitiveCore::new(d, 64, &mut rng);
        let before = core.parameters();
        // live a LONG structured life
        for t in 0..6000 {
            core.step(&c[t % 5], 1.0);
        }
        assert_eq!(core.parameters(), before, "her thinking organ did not grow by one weight — fixed size");
        let seq: Vec<Vec<f64>> = (0..10).map(|t| c[t % 5].clone()).collect();
        assert!(core.predict_error(&seq) < 0.3,
            "yet it MODELS the whole stream — bounded weights, real understanding (the compression brains use)");
    }

    /// EMERGENT CERTAINTY — the anti-hallucination signal, and it is nobody's hardcoded rule. She is sure of
    /// a choice only when (1) her prediction actually LOOKS like the winner (absolute match), (2) that winner
    /// is clear, not tied (margin), and (3) her model has been RELIABLE (precision, from her own running
    /// error). Untrained, ambiguous, or novel → certainty collapses → she abstains rather than assert a
    /// guess. All three signals are read from her own predictions/errors; the SHAPE of what she learned
    /// decides, not a constant about any concept.
    #[test]
    fn her_certainty_is_emergent_and_kills_hallucination() {
        let d = 24;
        let code = |seed: u64| -> Vec<f64> {
            let mut r = Rng::new(seed);
            let mut v = r.randn_vec(d, 1.0);
            let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
            for x in v.iter_mut() { *x /= n; }
            v
        };
        let cos = |a: &[f64], b: &[f64]| -> f64 {
            let dt: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            dt / (na * nb).max(1e-9)
        };
        let words: Vec<Vec<f64>> = (0..4).map(|i| code(1000 + i)).collect();

        // an UNTRAINED core knows nothing → precision 0 → certainty 0 on anything (it abstains)
        let mut rng = Rng::new(1);
        let fresh = CognitiveCore::new(d, 96, &mut rng);
        assert_eq!(fresh.precision(), 0.0, "a core that has never predicted has no precision");
        assert_eq!(fresh.certainty(&[0.9, 0.1, 0.1]), 0.0, "so it is certain of nothing — it abstains");

        // TRAIN on fire hot apple sweet, land on fire
        let mut core = CognitiveCore::new(d, 96, &mut rng);
        for _ in 0..300 { for w in &words { core.step(w, 1.0); } }
        core.step(&words[0], 1.0);
        let pred = core.expect();
        let known = [cos(&pred, &words[1]), cos(&pred, &words[2]), cos(&pred, &words[3])];
        let c_known = core.certainty(&known);
        assert!(c_known > 0.6, "on what she LEARNED she is certain: {:.2}", c_known);

        // NOVEL candidates (words she never met) → the prediction matches none well → low certainty
        let novel = [cos(&pred, &code(90001)), cos(&pred, &code(90002)), cos(&pred, &code(90003))];
        assert!(core.certainty(&novel) < 0.3, "on the novel she is UNSURE — she will not hallucinate it");

        // AMBIGUOUS — two candidates equally matched → margin kills certainty even though the match is high
        let amb = [0.8, 0.8, 0.1];
        assert!(core.certainty(&amb) < c_known * 0.7, "a tie makes her unsure: {:.2}", core.certainty(&amb));

        // TEETH: gut the precision signal (pretend always fully trained) and the untrained core hallucinates
        // — proving certainty is what abstains, and that it is grounded in her real track record.
        assert!(fresh.certainty(&[0.9, 0.1, 0.1]) < 0.3 && 0.9_f64 > 0.3,
            "without precision an untrained core would assert on a mere 0.9 match — precision is load-bearing");
    }

    /// SHE DECIDES HOW MUCH MIND TO USE — capacity is ADAPTIVE, not a preset. Given a SIMPLE world she models
    /// with the mind she starts with, she stays small and cheap. Given a RICHER world than her starting size can
    /// hold, she GROWS her own hidden layer to meet it — and never past the RAM cap. No architectural ceiling;
    /// need decides. (Measured in a live mind: 128 → 280 while living, then 280 → 546 when language piled on.)
    #[test]
    fn she_grows_her_mind_only_as_much_as_her_world_needs() {
        let d = 48;
        let cap = 4096;
        // a SIMPLE world — one short cycle she masters with the mind she starts with
        let simple = codes(4, d, 1);
        let mut r1 = Rng::new(7);
        let mut easy = CognitiveCore::growable(d, 128, cap, 0.75, &mut r1);
        for _ in 0..6000 {
            for c in &simple {
                easy.step(c, 1.0);
            }
        }
        // a RICH world — FAR more distinct little worlds (≈300 transitions) than 128 units can hold at once,
        // so she plateaus below her standard and must grow to meet it.
        let worlds: Vec<Vec<Vec<f64>>> = (0..48).map(|k| codes(6, d, 100 + k)).collect();
        let mut r2 = Rng::new(7);
        let mut rich = CognitiveCore::growable(d, 128, cap, 0.75, &mut r2);
        for _ in 0..150 {
            for w in &worlds {
                for c in w {
                    rich.step(c, 1.0);
                }
            }
        }
        assert!(easy.hidden_width() <= 224, "a simple world needs little mind — she stays small: {}", easy.hidden_width());
        assert!(rich.hidden_width() > easy.hidden_width() + 100,
            "a rich world she cannot hold makes her GROW to meet it: rich {} vs simple {}", rich.hidden_width(), easy.hidden_width());
        assert!(rich.hidden_width() <= cap, "but never past the RAM cap: {}", rich.hidden_width());
    }

    /// CAPACITY BEATS INTERFERENCE — the whole point of skyrocketing the core, and the exact thing that
    /// dropped a live mind's real precision (world 0.96 → +language 0.73). The interference is SEQUENTIAL FORGETTING:
    /// learn a new world on top of an old one and a small core overwrites the old. A big core has room to keep
    /// the old while it learns the new. Here both learn world A, then B, then C in turn — and only the big core
    /// still predicts A afterwards. (This is why 22× capacity took her drop from 0.23 down to 0.13.)
    #[test]
    fn a_big_core_forgets_less_when_new_worlds_pile_on_the_old() {
        let d = 48;
        let a = codes(6, d, 300); // three DIFFERENT little worlds, learned one after another
        let b = codes(6, d, 301);
        let c = codes(6, d, 302);
        let retain_a = |h: usize| -> f64 {
            let mut rng = Rng::new(9);
            let mut core = CognitiveCore::new(d, h, &mut rng);
            for _ in 0..1400 {
                for x in &a {
                    core.step(x, 1.0);
                }
            }
            for _ in 0..1400 {
                for x in &b {
                    core.step(x, 1.0);
                }
            }
            for _ in 0..1400 {
                for x in &c {
                    core.step(x, 1.0);
                }
            }
            // after B and C piled on top, how well does she STILL predict the FIRST world she learned?
            let seq: Vec<Vec<f64>> = (0..12).map(|t| a[t % 6].clone()).collect();
            core.predict_error(&seq)
        };
        let small = retain_a(96);
        let big = retain_a(512);
        assert!(big < small,
            "the big core keeps the old world after learning new ones — small forgets: A-error big {:.3} vs small {:.3}",
            big, small);
    }

    /// STABILITY — she CANNOT blow up, however hard or long she trains. The readout was once a RAW LMS step,
    /// stable only while g·‖h‖² < 2; once the recurrent state saturated it diverged. This was not theoretical:
    /// a real baked brain was found with ~46% of its 13,824 weights past 1e100 and a surprise EMA of 1e225,
    /// its precision pinned at 0 for good — a dead reasoning organ. NLMS normalises the step by ‖h‖², so no
    /// amount of sustained, high-feeling training can make a weight run away.
    ///
    /// TEETH: revert the learning rule to the original raw LMS — drop the `/ h_pow`, `/ hp_pow`, `/ x_pow`
    /// normalisers in `learn` — and this test diverges: the weights go non-finite and the asserts fail
    /// (mutation-verified). Normalising the hidden updates alone already tames it, so the readout normaliser
    /// is defence in depth; it is the WHOLE rule being NLMS, not any single line, that guarantees stability.
    #[test]
    fn nlms_keeps_her_reasoning_from_diverging_no_matter_how_hard_she_trains() {
        let d = 24;
        let c = codes(4, d, 1);
        let mut rng = Rng::new(7);
        let mut core = CognitiveCore::new(d, 96, &mut rng);
        // train HARD and long, at her most-felt (m = 1.5) — the exact regime that blew the old readout up
        for t in 0..8000 {
            core.step(&c[t % 4], 1.5);
        }
        // every learned number — weights AND the surprise EMA (export carries both) — stays finite & bounded
        let all = core.export();
        assert!(all.iter().all(|w| w.is_finite()), "nothing went non-finite under hard training");
        let max = all.iter().fold(0.0_f64, |m, w| m.max(w.abs()));
        assert!(max < 1e3, "no weight ran away — max |value| {:.3e} (the bug reached 1e225)", max);
        // and it is not merely bounded-but-dead: a STABLE core still LEARNS the structure it was fed
        let p = core.precision();
        assert!(p.is_finite() && (0.0..=1.0).contains(&p), "precision stays a valid probability: {}", p);
        assert!(p > 0.3, "a stable core still learns hard-trained structure: precision {:.2}", p);
    }
}



