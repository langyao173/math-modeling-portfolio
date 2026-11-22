# Ceiling and B002 Frontier Audit V33

## Purpose

The user target was to push toward `0.9999` or beyond the previously reported
theoretical bound. This audit checks whether that target is legal under the
current evaluator and whether B002 has a remaining high-yield physical placement
frontier.

## Legal Ceiling Finding

The current evaluator injects final readout noise on every measured qubit:

- Track A: `cirq.bit_flip(readout_error)` immediately before final measurement.
- Track B: Stim `X_ERROR(readout_error)` immediately before `M`.

The validator/equivalence path also requires exactly one final `z` measurement
whose width matches the logical program width. Therefore a legal compiler cannot
remove final readout noise. A literal `0.9999` public score is not reachable
under the current public backend snapshots unless the evaluator contract changes
or a validator/simulator bug is found.

Existing rule-aware readout-only public cap:

| Component | Cap |
|---|---:|
| Track A public average | 0.9506936060 |
| Track B public average | 0.8101493709 |
| Weighted public final | 0.9155575473 |

This is an optimistic cap with zero gate noise. It is not a constructive
guarantee.

## B002 Frontier Search

Diagnostic script:

`experiments/search_b002_star_frontier.py`

Searches performed:

| Run | Layouts | Simulated | Shots | Result |
|---|---:|---:|---:|---|
| `b002_star_frontier_search_v2_small.json` | 30 | 12 | 1024 | best mean 0.634765625 |
| `b002_star_frontier_search_v3_wide.json` | 80 | 20 | 1024 | best mean 0.6376953125 |

Best wide candidate:

- name: `offline_star_031_readout_copyout`
- CX: 19
- depth: 6
- estimated log reliability: `-0.5523563925922755`
- readout cost: `0.29001822893429613`
- sensitivity: `0.45958750000000004`
- five-seed 1024-shot mean: `0.6376953125`

The wider search did not find a new metric frontier. Top candidates are
permutations of the same six 4-qubit parity-star groups plus one readout
copyout. They share the same ELR/readout profile; observed differences are
consistent with finite-shot variance.

## Structural Interpretation

B002 public parity consists of six independent 4-bit repetition-code cosets.
Each group has one random root and three leaves constrained by parity checks.
Producing this valid support without duplicate measurement or classical
post-processing requires correlating the four output bits. Under the allowed
unitary/final-measurement model, that requires at least three entangling links
per 4-bit component, hence the current 18-CX star is already at the natural
component lower bound. The 19-CX copyout trades one extra CX for lower readout
on a high-error output.

## Decision

No runtime promotion from this audit.

The legal high-ceiling objective remains approaching the `~0.916` readout-only
public cap, not `0.9999`. The B002 physical placement frontier appears saturated
for the current affine-star/copyout family.
