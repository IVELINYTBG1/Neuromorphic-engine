# Neuromorphic-engine

A bio-faithful **neuromorphic substrate** written in native Rust with **zero dependencies** (std only): a
network of biologically-modelled cells that learn with purely **local, backprop-free, neuromodulator-gated**
rules on the CPU. No training loop over a frozen corpus, no gradient descent, no injection — every cell learns
online, from its own stream, the way a brain does.

Each `src/bio_*.rs` module is one self-contained, tested cell (a docstring explaining the biology → math, plus
a `#[cfg(test)]` self-test). **66 cells, 75 tests green.**

## What's inside (the families)

- **Learning principles** — LIF membrane + spike-frequency adaptation + k-WTA sparsity + three-factor local
  learning + neuromodulator-warped excitability (`bio_layer`); STDP timing (`bio_stdp`); direct feedback
  alignment, no weight transport (`bio_net`); ternary/1-bit weights (`bio_ternary`); delta/event coding
  (`bio_delta`); polychronization, branching, noise-tolerant population codes, conjunctive cells
  (`bio_poly`, `bio_branch`, `bio_noise`, `bio_conj`).
- **Memory** — NMDA gate, calcium control, CaMKII bistable switch, synaptic tagging & capture, working memory,
  engram attractors, sleep-replay consolidation, reconsolidation, and **structural plasticity** (memory as
  new pathways: synaptogenesis / pruning / neurogenesis — `bio_structural`).
- **Thinking dynamics** — E/I balance, heteroclinic trajectories, oscillatory binding, basal-ganglia
  selection, predictive coding, global-workspace ignition.
- **Sensing** — transduction, center-surround, simple→complex hierarchy, thalamic gating, Bayesian fusion,
  efference copy, interoception, spatial (place + grid) cells.
- **Limbic / value** — reward-prediction-error (dopamine), fear conditioning (amygdala), core affect, drive,
  somatic markers, and one diffuse neuromodulator that reconfigures the whole network.
- **Motor** — population-vector cortex, CPG, cerebellar forward model, stretch reflex, adaptation,
  equilibrium-point control.
- **Endocrine / glia** — hormones as a slow neuromodulator table; astrocytes (K⁺ buffering, Ca²⁺ waves,
  D-serine gating, lactate shuttle).
- **Higher cells** — a **grown cortex** where a concept is a sparse population and pathways are built by
  living (`bio_cortex`); a **learned predictive-recurrent reasoning core** — online, local, feeling-gated,
  with an emergent certainty/precision signal (`bio_cognition`); grounded semantics (`bio_semantics`),
  grammar (`bio_grammar`), a tiny action-DSL (`bio_code`) and sandboxed effectors (`bio_node`); plus two
  integration capstones (`bio_organism`, `bio_being`).

## Build & test

```
cargo build --release
cargo test --release      # 75 tests
```

## License

MIT.
