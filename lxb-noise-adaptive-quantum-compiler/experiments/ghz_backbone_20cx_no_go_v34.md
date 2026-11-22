# GHZ Backbone 20-CX No-Go V34

## Question

B001 remains the largest promoted-root headroom sample. The current selected
GHZ leaf-tree candidate uses 21 CX. A graph search found connected six-node
internal backbones with 15 external leaves, which can realize a clean-leaf GHZ
measurement channel with 20 CX.

## Result

The 20-CX candidates are valid but worse under noisy simulation than the current
21-CX selected candidate.

Five-seed 4096-shot B001 diagnostic:

| candidate | CX | depth | ELR | mean success | min success |
|---|---:|---:|---:|---:|---:|
| selected current tree | 21 | 11 | -0.5020120329 | 0.66318359375 | 0.656494140625 |
| best 20-CX backbone | 20 | 14 | -0.5102038762 | 0.65361328125 | 0.650634765625 |
| second 20-CX backbone | 20 | 14 | -0.5110352899 | 0.65361328125 | 0.647216796875 |
| shallow 20-CX backbone | 20 | 11 | -0.5232035453 | 0.64423828125 | 0.635009765625 |

## Interpretation

Reducing B001 from 21 CX to 20 CX is possible structurally, but the required
compact backbone uses worse edges/readout and loses more reliability than the
one-CX saving recovers.

## Decision

Do not promote the 20-CX backbone generator into runtime. The attempted runtime
generator was reverted because it increased B001 compile time without improving
selected score.
