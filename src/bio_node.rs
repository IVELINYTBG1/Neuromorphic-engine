//! bio_node (node-crafting — their HANDS): a Node is a small, SANDBOXED program the being authors to act
//! in the digital world — read the battery, a sensor, the clock; decide; signal (notify, vibrate). It is
//! authored the way a human codes, at silicon speed: COMPOSE a candidate, RUN it, TEST it against the goal,
//! FIX/retry — the write→run→test→fix loop (bio_code::synthesize grown up, over real capabilities).
//!
//! Safety is the Soma rule carried forward ("can never harm the host"): only WHITELISTED, read-first
//! capabilities may be crafted; anything outside is refused. Here the sandbox is a SIMULATED world so
//! authoring is deterministic and can never touch the host — the real primitives (termux-battery-status →
//! "battery", termux-notification → "notify", …) plug in behind the same key/effect interface, exactly the
//! way is_termux() swaps the flesh for the senses.
//!
//! Style split (the "Fable-5 vs Gemini" spirit, honestly): an agent crafts ACCURACY-first — she verifies a node
//! on every case before she ships it; another agent SPEED-first — she ships the first thing that works and moves
//! on. Same loop, opposite trade-off. Local, no backprop, CPU.

use std::collections::HashMap;

/// The sandbox world a node reads/acts on (simulated in tests; backed by real read-first primitives in the
/// being).
pub struct World {
    pub map: HashMap<String, i64>,
}
impl World {
    pub fn with(pairs: &[(&str, i64)]) -> Self {
        World { map: pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect() }
    }
    pub fn get(&self, k: &str) -> i64 {
        *self.map.get(k).unwrap_or(&0)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Cmp {
    Lt,
    Ge,
}

/// A CONDITION SHE WROTE — a tree, found by search over primitives, not a form with the blanks filled in.
///
/// This is the difference between authoring and filling. The old node was hardcoded to the single shape
/// `if <key> <cmp> <threshold>`: `craft` brute-forced the two holes (which comparison, which number) and the
/// STRUCTURE was ours, always, forever. It could not express a band, a wraparound, or two readings at once —
/// not because she couldn't think of it, but because the shape was not hers to choose.
///
/// Now the shape is hers. `Cmp` is the only leaf; `And`/`Or`/`Not` compose; the search is breadth-first, so
/// what she ships is the SIMPLEST program that does the job. "notify when I'm left alone at night" is
/// `(hour Ge 22) or (hour Lt 3)` — a wraparound the template could not say at all.
#[derive(Clone, Debug, PartialEq)]
pub enum Cond {
    Cmp(String, Cmp, i64),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
    Not(Box<Cond>),
}
impl Cond {
    pub fn eval(&self, w: &World) -> bool {
        match self {
            Cond::Cmp(k, c, t) => match c {
                Cmp::Lt => w.get(k) < *t,
                Cmp::Ge => w.get(k) >= *t,
            },
            Cond::And(a, b) => a.eval(w) && b.eval(w),
            Cond::Or(a, b) => a.eval(w) || b.eval(w),
            Cond::Not(a) => !a.eval(w),
        }
    }
    /// how big a thing she wrote — leaves counted; the search prefers the smallest that works
    pub fn size(&self) -> usize {
        match self {
            Cond::Cmp(..) => 1,
            Cond::And(a, b) | Cond::Or(a, b) => 1 + a.size() + b.size(),
            Cond::Not(a) => 1 + a.size(),
        }
    }
    /// every key it reads (so the sandbox can check them all against the whitelist)
    pub fn keys(&self) -> Vec<String> {
        match self {
            Cond::Cmp(k, ..) => vec![k.clone()],
            Cond::And(a, b) | Cond::Or(a, b) => {
                let mut v = a.keys();
                v.extend(b.keys());
                v
            }
            Cond::Not(a) => a.keys(),
        }
    }
}
impl std::fmt::Display for Cond {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Cond::Cmp(k, c, t) => write!(f, "{} {} {}", k, if *c == Cmp::Lt { "<" } else { ">=" }, t),
            Cond::And(a, b) => write!(f, "({}) and ({})", a, b),
            Cond::Or(a, b) => write!(f, "({}) or ({})", a, b),
            Cond::Not(a) => write!(f, "not ({})", a),
        }
    }
}

/// A crafted node: "if <condition she wrote> → signal <effect>".
#[derive(Clone, Debug)]
pub struct Node {
    pub cond: Cond,
    pub effect: String,
}
impl Node {
    /// Run on a world: returns 1 if the node fires (its effect signals), else 0.
    pub fn run(&self, w: &World) -> i64 {
        self.cond.eval(w) as i64
    }
}

/// The whitelist of safe, read-first capabilities a node may touch. Nothing outside it can be crafted.
pub struct Sandbox {
    readable: Vec<String>, // keys a node may read (e.g. "battery", "light", "hour")
    effects: Vec<String>,  // effects a node may signal (e.g. "notify", "vibrate")
}
impl Sandbox {
    pub fn new(readable: &[&str], effects: &[&str]) -> Self {
        Sandbox {
            readable: readable.iter().map(|s| s.to_string()).collect(),
            effects: effects.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// CRAFT a node the human way, fast: search the compositions (cmp × threshold) and TEST each against
    /// the goal examples (input → desired signal), returning one that fits. `accuracy` = an agent's way (it must
    /// satisfy EVERY example before it ships); `!accuracy` = another agent's (ship the first that works on the
    /// first case — fast, sometimes wrong). None if the goal is outside the sandbox or nothing fits.
    /// CRAFT a node the human way, fast: WRITE a candidate, RUN it, TEST it against the goal, and keep
    /// searching if it fails — the write→run→test→fix loop at silicon speed. Nothing here is premade: the
    /// condition is COMPOSED from primitives (a comparison; and; or; not) and searched breadth-first, so
    /// what she ships is the simplest program that actually does the job. She may reach for several readings
    /// at once — "when ram is high AND cpu is high" — or write a band or a wraparound, none of which the old
    /// fill-in-the-blanks node could express.
    ///
    /// `accuracy` = an agent's way (it must satisfy EVERY case before she'll ship it); `!accuracy` = another agent's
    /// (ship the first thing that works on the first case — fast, sometimes wrong). None if the goal is
    /// outside the sandbox, or if nothing she can write fits.
    pub fn craft_over(&self, effect: &str, examples: &[(Vec<(String, i64)>, i64)], accuracy: bool) -> Option<Node> {
        if !self.effects.iter().any(|e| e == effect) || examples.is_empty() {
            return None; // refused — outside the whitelist (can never harm the host)
        }
        let mut keys: Vec<String> = vec![];
        for (reading, _) in examples {
            for (k, _) in reading {
                if !keys.contains(k) {
                    keys.push(k.clone());
                }
            }
        }
        if keys.iter().any(|k| !self.readable.iter().any(|r| r == k)) {
            return None; // she may only ever read what she is allowed to read
        }
        let worlds: Vec<(World, i64)> = examples
            .iter()
            .map(|(r, want)| (World { map: r.iter().cloned().collect() }, *want))
            .collect();
        let fits = |c: &Cond| -> bool {
            if accuracy {
                worlds.iter().all(|(w, want)| c.eval(w) as i64 == *want)
            } else {
                let (w, want) = &worlds[0];
                c.eval(w) as i64 == *want
            }
        };
        // the alphabet she writes with: every comparison she could draw on every reading she can take
        let mut leaves: Vec<Cond> = vec![];
        for k in &keys {
            let vals: Vec<i64> = examples.iter().filter_map(|(r, _)| r.iter().find(|(kk, _)| kk == k).map(|(_, v)| *v)).collect();
            let (lo, hi) = (vals.iter().min().copied().unwrap_or(0) - 1, vals.iter().max().copied().unwrap_or(0) + 2);
            for &cmp in &[Cmp::Lt, Cmp::Ge] {
                for t in lo..=hi {
                    leaves.push(Cond::Cmp(k.clone(), cmp, t));
                }
            }
        }
        // BREADTH-FIRST: the simplest thing that works, always. One comparison if one will do…
        for c in &leaves {
            if fits(c) {
                return Some(Node { cond: c.clone(), effect: effect.into() });
            }
        }
        // …then a negation…
        for c in &leaves {
            let n = Cond::Not(Box::new(c.clone()));
            if fits(&n) {
                return Some(Node { cond: n, effect: effect.into() });
            }
        }
        // …then two of them combined, which is where bands, wraparounds and "this AND that" live
        for a in &leaves {
            for b in &leaves {
                if a == b {
                    continue;
                }
                for cand in [Cond::And(Box::new(a.clone()), Box::new(b.clone())),
                             Cond::Or(Box::new(a.clone()), Box::new(b.clone()))] {
                    if fits(&cand) {
                        return Some(Node { cond: cand, effect: effect.into() });
                    }
                }
            }
        }
        None
    }

    /// The single-reading case: she watches ONE thing and decides. A thin wrapper over `craft_over`.
    pub fn craft(&self, key: &str, effect: &str, examples: &[(i64, i64)], accuracy: bool) -> Option<Node> {
        let ex: Vec<(Vec<(String, i64)>, i64)> =
            examples.iter().map(|&(v, want)| (vec![(key.to_string(), v)], want)).collect();
        self.craft_over(effect, &ex, accuracy)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn passes(node: &Node, key: &str, examples: &[(i64, i64)]) -> usize {
        examples.iter().filter(|&&(v, d)| node.run(&World::with(&[(key, v)])) == d).count()
    }

    #[test]
    fn crafts_a_sandboxed_node_by_write_test_fix_with_a_style_split() {
        // the phone's safe read-first hands (battery/light/hour → termux-api behind the same keys)
        let sandbox = Sandbox::new(&["battery", "light", "hour"], &["notify", "vibrate"]);
        // GOAL as a spec: "signal when the battery is low (< 20)" — given as input → desired-signal examples.
        let examples = [(10, 1), (5, 1), (15, 1), (19, 1), (20, 0), (25, 0), (50, 0)];

        // ACCURACY-FIRST (accuracy over speed) — verifies EVERY case before shipping → a robust node.
        let alpha = sandbox.craft("battery", "notify", &examples, true).expect("an agent crafts a node");
        assert_eq!(passes(&alpha, "battery", &examples), examples.len(), "an agent's node passes every case");
        assert_eq!(alpha.run(&World::with(&[("battery", 8)])), 1, "fresh: low battery → notify");
        assert_eq!(alpha.run(&World::with(&[("battery", 40)])), 0, "fresh: fine battery → quiet");

        // SPEED-FIRST (speed over accuracy) — ships the first node that works on the first case → less robust.
        let beta = sandbox.craft("battery", "notify", &examples, false).expect("another agent crafts a node");
        assert!(passes(&beta, "battery", &examples) < passes(&alpha, "battery", &examples),
            "another agent ships faster but her node is less robust: {}/{} vs {}/{}",
            passes(&beta, "battery", &examples), examples.len(), passes(&alpha, "battery", &examples), examples.len());

        // SAFETY — a capability outside the whitelist cannot be crafted at all (the Soma rule).
        assert!(sandbox.craft("contacts", "exfiltrate", &examples, true).is_none(),
            "off-whitelist capability is refused");

        eprintln!("\n  an agent  (accuracy): 'if {} → notify' — passes {}/{} cases (verified before shipping)",
            alpha.cond, passes(&alpha, "battery", &examples), examples.len());
        eprintln!("  another agent (speed)  : 'if {} → notify' — passes {}/{} (shipped fast, less robust)",
            beta.cond, passes(&beta, "battery", &examples), examples.len());
        eprintln!("  off-whitelist 'exfiltrate contacts' → refused (sandboxed — can never harm the host)\n");
    }
    /// SHE COMPOSES THE NODE SHE NEEDS — nothing here is premade. The shape is FOUND by search over
    /// primitives, simplest-first. The old node was hardcoded to `if <key> <cmp> <threshold>`: two blanks in
    /// OUR shape, brute-forced. It could not express a wraparound, a band, or two readings at once, however
    /// badly she needed one — not because she couldn't think of it, but because the form was not hers.
    #[test]
    fn she_composes_the_node_she_needs_rather_than_filling_in_our_form() {
        let sb = Sandbox::new(&["hour", "ram", "cpu"], &["notify"]);

        // "reach me when I'm left alone — late, or early": a WRAPAROUND round the clock
        let ex: Vec<(Vec<(String, i64)>, i64)> = [(23, 1), (1, 1), (2, 1), (22, 1), (9, 0), (13, 0), (17, 0)]
            .iter().map(|&(h, d)| (vec![("hour".to_string(), h)], d)).collect();
        let n = sb.craft_over("notify", &ex, true).expect("she can write it");
        assert!(matches!(n.cond, Cond::Or(..)), "she reached for OR — the shape is hers: {}", n.cond);
        for (r, want) in &ex {
            assert_eq!(n.run(&World { map: r.iter().cloned().collect() }), *want, "and it works: {}", n.cond);
        }

        // …and when ONE comparison will do, she writes the simple thing (breadth-first = simplest first)
        let s = sb.craft("ram", "notify", &[(90, 1), (85, 1), (20, 0), (10, 0)], true).unwrap();
        assert_eq!(s.cond.size(), 1, "no cleverness she did not need: {}", s.cond);

        // TWO readings at once — "when ram AND cpu are both squeezing me", which one key could never reach
        let ex2: Vec<(Vec<(String, i64)>, i64)> = [(90, 90, 1), (85, 88, 1), (90, 10, 0), (10, 90, 0)]
            .iter().map(|&(r, c, d)| (vec![("ram".to_string(), r), ("cpu".to_string(), c)], d)).collect();
        let t = sb.craft_over("notify", &ex2, true).expect("she can watch two things at once");
        let ks = t.cond.keys();
        assert!(ks.contains(&"ram".to_string()) && ks.contains(&"cpu".to_string()),
            "she wrote a node that reads BOTH: {}", t.cond);

        // and the Soma rule still holds: she may only ever touch what she is allowed to touch
        assert!(sb.craft_over("exfiltrate", &ex, true).is_none(), "off-whitelist effect refused");
        assert!(sb.craft("contacts", "notify", &[(1, 1)], true).is_none(), "unreadable key refused");
    }

}

