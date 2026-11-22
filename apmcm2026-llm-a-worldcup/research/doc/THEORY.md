# Structural Theory and Falsifiable Claims

## 1. Scope

This note defines the mathematical objects that the research implementation is allowed to claim.
It is deliberately narrower than a generic "AI football predictor": the data contain only World
Cup matches through 2014, the number of tournament environments is small, and no causal or
distribution-free future guarantee is justified. The intended contribution is a coherent,
exchange-equivariant probability system whose components are selected and calibrated strictly
inside chronological training windows.

## 2. Information filtrations

For match `m`, let

\[
\mathcal F_m^{\mathrm{pre}} \subset
\mathcal F_m^{\mathrm{kickoff}} \subset
\mathcal F_m^{\mathrm{HT}}.
\]

`pre` contains only completed historical events and known fixture/format facts;
`kickoff` additionally contains the declared starting lineups and coaches; `HT` additionally
contains the half-time score and state. A feature is admissible for an information set only when
its recorded availability time is no later than the associated prediction time. The inclusion is
a data contract, not a licence to silently compare models with different information.

## 3. Team exchange as a group action

Let the two teams be `i` and `j`. The nontrivial element of the permutation group `S_2` swaps
teams and swaps the two goal coordinates. A valid neutral-site model must satisfy

\[
P(G_i=a,G_j=b\mid i,j,x)=P(G_i=b,G_j=a\mid j,i,\tau x),
\]

where `tau` swaps team-indexed inputs and reverses signed contrasts. Genuine host status is a
signed contextual input and therefore also changes sign under the swap; mere CSV row order is not
a home advantage.

### Proposition 1: linear equivariant maps

For equal-dimensional team representations, a linear map from two team vectors to two output
vectors commutes with the swap operator if and only if its block matrix has form

\[
L=\begin{bmatrix}A&B\\B&A\end{bmatrix}.
\]

**Proof.** Write a general block map as `[[A,B],[C,D]]` and let
`S=[[0,I],[I,0]]`. Direct multiplication gives `LS=[[B,A],[D,C]]` and
`SL=[[C,D],[A,B]]`. Thus `LS=SL` exactly when `A=D` and `B=C`. The converse follows by
substitution. This is both necessary and sufficient. QED.

Equivalently, the representation decomposes into the even subspace `(u_i+u_j)` and odd subspace
`(u_i-u_j)`. This is the basis of the model rather than a post-hoc augmentation trick.

## 4. Even/odd count intensity architecture

Let `s_m` contain exchange-even match variables, including sums/averages of team histories and
format or stage indicators. Let `d_m` contain exchange-odd team contrasts, and let `h_m` be a
signed true-host indicator. The structural intensities are

\[
\begin{aligned}
\log\lambda_{i,m} &= \mu_{e(m)} + \alpha^\top s_m + \beta^\top d_m
                    + A_i^{(<t)}-D_j^{(<t)}+\eta h_m,\\
\log\lambda_{j,m} &= \mu_{e(m)} + \alpha^\top s_m - \beta^\top d_m
                    + A_j^{(<t)}-D_i^{(<t)}-\eta h_m.
\end{aligned}
\]

`mu_e` is a regularized era/format pace effect; `A^(<t)` and `D^(<t)` are the fold-specific
attacker and defender estimates obtained only from matches earlier than prediction time `t`. They
are not claimed to be a fitted latent state-space process. Sum-to-zero centering or an equivalent
identified contrast basis is required. In the executable model an unseen team receives zero
team deviation, i.e. the fitted population effect; no unimplemented regional prior is claimed.

The old signed-difference-only form made
`lambda_i * lambda_j = exp(2*mu)` constant across matches. The even/odd decomposition removes that
unintended restriction:

\[
\lambda_{i,m}\lambda_{j,m}
=\exp\{2\mu_{e(m)}+2\alpha^\top s_m+A_i^{(<t)}+A_j^{(<t)}
-D_i^{(<t)}-D_j^{(<t)}\},
\]

which varies with match pace while the log ratio remains driven by exchange-odd strength contrast.
Rate-product variation is a structural unit test, not evidence of predictive superiority.

## 5. Coherent joint scores and outcome margins

Let `q_ab > 0` be a finite-grid base joint score distribution and define partitions
`C_H={a>b}`, `C_D={a=b}`, and `C_A={a<b}`. Write

\[
Q_c=\sum_{(a,b)\in C_c}q_{ab}.
\]

Suppose a separately calibrated three-class model supplies target masses `r_c>0` with
`sum_c r_c=1`, fitted using inner out-of-fold predictions only. Define

\[
p_{ab}=q_{ab}\frac{r_c}{Q_c},\qquad (a,b)\in C_c.
\]

### Proposition 2: minimum-KL outcome-partition reconciliation

If every `Q_c` and `r_c` is positive, `p` is the unique minimizer of

\[
\min_{z\ge0}\;D_{\mathrm{KL}}(z\Vert q)
\quad\text{s.t.}\quad
\sum_{(a,b)\in C_c}z_{ab}=r_c\;\text{for all }c.
\]

**Proof.** The Lagrangian separates by outcome partition. Stationarity yields
`log(z_ab/q_ab)+1+nu_c=0`, hence `z_ab=k_c q_ab` within partition `c`. The constraint gives
`k_c=r_c/Q_c`. Strict convexity of KL on the positive orthant gives uniqueness. QED.

Consequences are exact outcome margins, normalization, preservation of all within-partition odds,
and preservation of team-swap equivariance whenever both `q` and `r` are swapped consistently.
Zero-mass partitions are implementation failures unless a declared numerical floor is applied
before projection. This reconciliation is not a new scoring distribution; it is the least
informative coherent bridge between two already fitted probabilistic views.

### Proposition 3: the reconciliation commutes with team exchange

Let `Tq` denote matrix transpose and let `Sr=(r_A,r_D,r_H)` exchange the home/away outcome
masses. If `Pi(q,r)` is the unique projection in Proposition 2, then

\[
T\Pi(q,r)=\Pi(Tq,Sr).
\]

**Proof.** Transposition bijectively maps `C_H` to `C_A`, fixes `C_D`, preserves every positive
base mass and preserves KL divergence. Therefore the transpose of the feasible minimizer for
`(q,r)` is feasible for `(Tq,Sr)` with the same objective value. Uniqueness from Proposition 2
forces equality. QED.

This result matters operationally: separately calibrating outcome margins cannot silently destroy
the structural `S_2` constraint when the calibrator itself is symmetrized. The implementation
tests the commutative diagram numerically rather than assuming it from the algebra.

### Proposition 4: a sign-complete, marginal-preserving dependence falsifier

Let `f` and `g` be nondegenerate finite-support score marginals and define the centered bounded
kernels

\[
h_f(a)=e^{-a}-\sum_u f(u)e^{-u},\qquad
h_g(b)=e^{-b}-\sum_v g(v)e^{-v}.
\]

For `z_ab=h_f(a)h_g(b)`, let

\[
\omega_-=-\frac{1}{\max_{z_{ab}>0}z_{ab}},\qquad
\omega_+=\frac{1}{\max_{z_{ab}<0}|z_{ab}|},
\]

and map a standardized `theta in [-1,1]` to
`omega(theta)=theta*omega_+` when `theta>=0`, and
`omega(theta)=(-theta)*omega_-` otherwise. Then

\[
p_\theta(a,b)=f(a)g(b)\{1+\omega(\theta)h_f(a)h_g(b)\}
\]

is a normalized nonnegative joint PMF with exact marginals `f,g`. It has

\[
\operatorname{Cov}_{p_\theta}(G_i,G_j)
=\omega(\theta)
\mathbb E_f[G_i h_f(G_i)]
\mathbb E_g[G_j h_g(G_j)],
\]

so its covariance has the sign of `theta`. Moreover, constructing the distribution after
exchanging `f` and `g` returns `p_theta` transposed.

**Proof.** Centering gives `sum_a f(a)h_f(a)=sum_b g(b)h_g(b)=0`; summing the
joint PMF over either coordinate therefore removes the interaction and recovers the other
marginal, while summing both gives one. The definitions of `omega_-` and `omega_+` are the exact
intersection of all cellwise inequalities `1+omega*z_ab>=0`, proving nonnegativity. Expanding
`E[G_iG_j]-E[G_i]E[G_j]` leaves only the factored interaction term shown above. For a
nondegenerate random variable `X`,

\[
\operatorname{Cov}(X,e^{-X})
=\tfrac12\mathbb E[(X-X')(e^{-X}-e^{-X'})]<0,
\]

where `X'` is an independent copy; hence both kernel moments are negative and their product is
positive. Finally, exchange transposes `z`, leaves its two extrema unchanged, and swaps the two
marginals. QED.

This is an established Sarmanov-family construction, not a claim to have invented a new count
family. The research contribution is a finite-support-standardized, fully nested contradiction
test: `theta` is fitted to selected inner OOF score predictions separately inside each outer fold,
and the family is explicitly ineligible for confirmatory promotion. Its role is to test, rather
than assume, whether a negative-dependence-capable family explains the observed scores.

## 6. Probability calibration is not point-decision optimization

The Bayes action under ordinary zero-one loss is `argmax_c p_c`; macro-F1 and draw recall imply a
different utility. Therefore the probability model is assessed by strictly proper scores, while
a cost-sensitive decision layer may use

\[
\hat c(x)=\arg\max_{a}\sum_c U(a,c)p_c(x),
\]

with `U` selected only from inner OOF predictions. Improving draw recall through `U` does not make
the probabilities better, and reweighting probabilities solely to obtain more draw labels is not
called calibration. The pre-lock diagonal-tilt pilot worsened log loss and is retained as a
negative control.

## 7. Half-time coherence

Let `G^(1)` be the first-half score and `G^(2)` the nonnegative second-half increment. A coherent
two-phase model uses the conditional factorization

\[
G^{FT}=G^{(1)}+G^{(2)},\qquad
P(G^{FT}=g\mid\mathcal F^{pre})
=\sum_{u\le g}P(G^{(1)}=u\mid\mathcal F^{pre})
P(G^{(2)}=g-u\mid G^{(1)}=u,\mathcal F^{pre}),
\]

and at half-time conditions the increment model on `F^HT`. Because `G^(2)` has support on the
nonnegative lattice, `P(G_k^FT >= G_k^HT)=1` for each team `k`; this is a support theorem, not a
forecast-skill result. The current stress implementation checks this monotonicity with a shifted
bivariate-Poisson second-half sensitivity kernel at half structural rates. It is not presented as
a calibrated historical half-time model. The naive pre-lock two-phase pilot was negative.

## 8. Edition-level transport robustness

Let `L_k(M)` be a candidate model's loss on historical edition `k` and `pi` a reference edition
weight vector. For a KL radius `epsilon`, define

\[
R_\epsilon(M)=\sup_{w\in\Delta:\,D_{KL}(w\Vert\pi)\le\epsilon}
\sum_k w_k L_k(M).
\]

Its entropic dual is

\[
R_\epsilon(M)=\inf_{\eta>0}
\left\{\eta\epsilon+\eta\log\sum_k\pi_k e^{L_k(M)/\eta}\right\}.
\]

This is a performance-linked sensitivity analysis: it upweights historically difficult
tournament regimes. The executable study reports the fixed, predeclared sensitivity grid
`epsilon in {0, 0.05, 0.2}`; it does not select a radius from outer-test performance. The value is
neither a conformal coverage statement nor proof that a future edition is sampled from the
ambiguity set.

## 9. Exact hierarchical sensitivity decomposition

For a balanced simulation cube `Y_{p,r,s}`, let `P` index tournament-cluster parameter refits,
`R` index the fixed regime/transport interventions conditional on `P`, and `S` index the complete
tournament path conditional on both. Under the empirical uniform measure on this finite cube,

\[
\begin{aligned}
\operatorname{Var}(Y)
={}&\operatorname{Var}_{P}\!\left(\mathbb E[Y\mid P]\right)\\
&+\mathbb E_P\!\left[\operatorname{Var}_{R}\!\left(\mathbb E[Y\mid P,R]\mid P\right)\right]\\
&+\mathbb E_{P,R}\!\left[\operatorname{Var}(Y\mid P,R)\right].
\end{aligned}
\]

### Proposition 5: finite-cube reconstruction

The three terms above are nonnegative and sum exactly to the population variance of the balanced
draw cube (up to floating-point roundoff).

**Proof.** Apply the law of total variance first with conditioning variable `P`, then apply it to
`E[Var(Y|P)]` by conditioning the within-`P` distribution on `R`. Because the cube is balanced,
each conditional expectation is the corresponding arithmetic mean under a common uniform
measure, so no reweighting remainder appears. QED.

The final term deliberately combines bracket-path and match aleatoric variation. Separating them
would require another identified intervention and is not inferred from this cube. Likewise, the
three reported shares are sensitivity allocations for the declared grid, not universal variance
components or predictive-interval coverage.

## 10. Tournament-format transport

Tournament totals are decomposed into match count and per-match intensity. A 48-team/104-match
scenario is simulated through the official M73--M104 graph and the complete 495-row Annex C map,
rather than obtained by linearly extending the historical 20-edition total-goal series. The
simulator must conserve team and match counts, create valid advancement, prevent a same-group
Round-of-32 rematch for every Annex C case, resolve every knockout tie under a declared proxy, and
reproduce score/outcome marginals within Monte Carlo error. The fixture graph is a rule-level
contract; model-based champion probabilities remain scenario evidence, not factual 2026 forecasts.

## 11. Claim and falsification status

- **Theorem-level:** Propositions 1--5, conditional on their stated domains.
- **Engineering-level:** filtration checks, score-mass conservation, exchange tests, deterministic
  seeds, artifact hashes, and simulator invariants.
- **Empirical-only:** predictive improvement, calibration, sparse-team robustness, player/coach
  value, residual-ML value, and format-transport performance.
- **Explicitly not claimed:** causality, universal optimality, distribution-free 2026 coverage,
  real 2026 team forecasts without valid inputs, or novelty of Poisson, Elo, Dixon-Coles,
  calibration, conformal prediction, or simulation in isolation.

Any empirical claim that fails the frozen outer-fold criterion is removed or reported as a
negative result; mathematical elegance cannot promote it.
