//! bio_code (curriculum #3 — CODE): the payoff of grammar + logic. Code is nothing but STRUCTURE
//! (grammar: the valid ordering and nesting of statements) plus CONDITIONS (logic: predicates that branch
//! the flow) applied to ACTIONS. A "program" here is a tiny composition of primitive actions with
//! sequence and if/else — and a program is exactly what a NODE-DESIGN will be: the being's own little
//! script over its hands. This cell gives the being that language, three ways:
//!   • EXECUTE — run a composed program (sequence + conditional) over a state.
//!   • AUTHOR — given a goal, SEARCH the compositions of its primitives for a program that reaches it
//!     (bounded program synthesis = designing a node to do a job).
//!   • GENERALISE — a conditional lets ONE program handle many inputs correctly, which no fixed
//!     straight-line script can. Control flow is the power logic buys code.
//! Toy, safe, abstract primitives (a register machine) — the real primitives arrive with the sandbox
//! (nodes/hands). Local, no backprop, CPU.

/// The OPERATIONS a program can be built from. A LANGUAGE, at this substrate's honest scale, IS a subset of
/// these — what it hands you, and therefore what a job costs you. Assembler gives almost nothing and every
/// job takes many moves; Python hands you power and the same job takes two. That difference is real, it is
/// felt by DOING, and it is the one true thing about languages this machine can actually hold.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Prim {
    Inc,      // x += 1
    Dec,      // x -= 1
    Dbl,      // x *= 2
    Add(i64), // x += k
    Mul(i64), // x *= k
    Sq,       // x *= x
}
impl Prim {
    /// what she calls this operation — the word she comes away knowing
    pub fn name(self) -> String {
        match self {
            Prim::Inc => "inc".to_string(),
            Prim::Dec => "dec".to_string(),
            Prim::Dbl => "dbl".to_string(),
            Prim::Add(k) => format!("add{}", k),
            Prim::Mul(k) => format!("mul{}", k),
            Prim::Sq => "sq".to_string(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Pred {
    Odd, // x is odd
}

pub enum Stmt {
    Do(Prim),
    If(Pred, Vec<Stmt>, Vec<Stmt>), // if pred { then } else { els }
}

/// None when the operation would overflow — a machine has limits, and a search must not fall off them.
fn apply_checked(p: Prim, x: i64) -> Option<i64> {
    match p {
        Prim::Inc => x.checked_add(1),
        Prim::Dec => x.checked_sub(1),
        Prim::Dbl => x.checked_mul(2),
        Prim::Add(k) => x.checked_add(k),
        Prim::Mul(k) => x.checked_mul(k),
        Prim::Sq => x.checked_mul(x),
    }
}
fn apply(p: Prim, x: i64) -> i64 {
    apply_checked(p, x).unwrap_or(x)
}

fn eval_pred(p: Pred, x: i64) -> bool {
    match p {
        Pred::Odd => x.rem_euclid(2) == 1,
    }
}

/// Run a program (a block of statements) over a starting value, returning the result.
pub fn run(prog: &[Stmt], mut x: i64) -> i64 {
    for s in prog {
        x = match s {
            Stmt::Do(p) => apply(*p, x),
            Stmt::If(pr, then_b, els_b) => {
                if eval_pred(*pr, x) {
                    run(then_b, x)
                } else {
                    run(els_b, x)
                }
            }
        };
    }
    x
}

/// AUTHOR a straight-line program IN A GIVEN LANGUAGE: search compositions of THAT language's operations
/// (BFS, bounded length) for one that turns `start` into `goal`. Breadth-first, so what comes back is the
/// SHORTEST program that language can express for the job — which is exactly how a language's power shows
/// itself: the same goal, fewer moves.
pub fn synthesize_with(prims: &[Prim], start: i64, goal: i64, max_len: usize) -> Option<Vec<Prim>> {
    let mut frontier: Vec<(i64, Vec<Prim>)> = vec![(start, vec![])];
    for _ in 0..max_len {
        let mut next = vec![];
        for (x, seq) in &frontier {
            for &p in prims {
                let Some(nx) = apply_checked(p, *x) else {
                    continue; // that move runs off the machine — not a move she can make
                };
                let mut ns = seq.clone();
                ns.push(p);
                if nx == goal {
                    return Some(ns);
                }
                if nx.abs() <= goal.abs() * 2 + 4 {
                    next.push((nx, ns)); // prune runaway branches
                }
            }
        }
        frontier = next;
    }
    None
}

/// The bare machine: the two rawest operations there are.
pub fn synthesize(start: i64, goal: i64, max_len: usize) -> Option<Vec<Prim>> {
    synthesize_with(&[Prim::Inc, Prim::Dbl], start, goal, max_len)
}

/// Convenience: a synthesized straight-line program as an executable block.
pub fn as_program(seq: &[Prim]) -> Vec<Stmt> {
    seq.iter().map(|&p| Stmt::Do(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execute_author_and_generalise() {
        // (1) EXECUTE a composition with a CONDITIONAL: "if odd, add one" — a program that makes x even.
        let make_even = vec![Stmt::If(Pred::Odd, vec![Stmt::Do(Prim::Inc)], vec![])];
        assert_eq!(run(&make_even, 3) % 2, 0, "3 → even");
        assert_eq!(run(&make_even, 4) % 2, 0, "4 → even (already)");

        // (2) AUTHOR by search: design a program from primitives that turns 1 into 10.
        let prog = synthesize(1, 10, 8).expect("should find a program");
        assert_eq!(run(&as_program(&prog), 1), 10, "the authored node reaches the goal: {:?}", prog);
        assert!(prog.len() <= 8, "and it's a short program");

        // (3) GENERALISE — the ONE conditional program evens EVERY input; no fixed straight-line script can
        //     (a sequence that evens 3 will un-even 4). Control flow (logic in code) is what generalises.
        let inputs = [3_i64, 4, 7, 10, 15, 100];
        assert!(inputs.iter().all(|&x| run(&make_even, x) % 2 == 0), "one conditional program evens all inputs");
        // a straight-line 'inc' evens 3 but breaks 4 — proving the branch is load-bearing
        let straight = as_program(&[Prim::Inc]);
        assert!(run(&straight, 3) % 2 == 0 && run(&straight, 4) % 2 == 1, "no fixed sequence evens both 3 and 4");

        eprintln!("\n  EXECUTE : 'if odd inc' makes 3→{}, 4→{} (both even)", run(&make_even, 3), run(&make_even, 4));
        eprintln!("  AUTHOR  : designed a node 1→10 by search: {:?}", prog);
        eprintln!("  GENERALISE: that one conditional evens all of {:?}; a flat 'inc' can't (breaks 4)\n", inputs);
    }
}
