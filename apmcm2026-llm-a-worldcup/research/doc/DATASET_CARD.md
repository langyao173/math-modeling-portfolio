# Dataset Card: Supplied FIFA World Cup Tables

## Intended use

The supplied tables support the three APMCM 2026 Problem A tasks: tournament-level trend and
2018 total-goal audit, match-level/half-time goal analysis, and 1930–2010 to 2014 outcome
forecasting. They are suitable for retrospective model comparison under chronological splits.
They are not sufficient by themselves for a factual 2026 team forecast, injury analysis,
causal attribution, betting use, or distribution-free future coverage.

## Immutable inputs

| Table | Raw rows | SHA-256 |
|---|---:|---|
| tournaments | 20 | `a66240afe2ce2d7df5b701eebf012052193b90f33e003c9874cd3df85941003e` |
| matches | 852 | `cb4e68bb7e080a84a82175151a2c1ce5ef4ef43fcb5111a3cf6c53f69bc92363` |
| players | 37,784 | `753a68af0cd4fa38b67c7883c138700dfb37544f1ed898b1d176a03f92748d67` |

Every run verifies these hashes before canonicalization. A hash mismatch is fatal; the pipeline
does not silently accept a revised dataset.

## Canonical tables

| Table | Rows | Main unit | Canonical content SHA-256 |
|---|---:|---|---|
| tournaments | 20 | World Cup edition, 1930–2014 | `4b3d57f99bc8ae208e10c204c2c7c750c0aa2c2faa4b62fae8e7f071470420c2` |
| matches | 836 | match | `1af79dc83cf493ad9f6b3561dd85f6212186e7bf2049205ad9cb1613e43a658c` |
| players | 37,048 | player–team–match event row | `282a2dfc785b6bc4a173d7ecfa263cdbb965ace78555860946d68c2410a80f17` |

Sixteen duplicate match rows were removed by exact `(RoundID, MatchID)` grouping and 736 exact
player rows were removed. No conflicting duplicate group was found. All 20 editions reconcile
exactly on match count, total goals, and attendance after canonicalization.

## Missingness and semantics

- One canonical match attendance is missing. Attendance is never silently imputed into a
  pre-match research feature; task-specific models declare their own preprocessing.
- `Event` is absent in 28,225 of 37,048 canonical player rows. This generally means no encoded
  event, not an unknown player record, so missing event text is treated as an empty event list.
- `Position` is absent in 33,030 player rows and is not used as a mandatory feature.
- Goals, half-time goals, team initials, coach names, lineup flags, match identifiers, dates, and
  stage labels are complete after canonicalization.
- Historical “home/away” is often row ordering rather than venue status. The research model uses
  a separately derived host-team indicator and an exchange-symmetry contract.
- Penalty shoot-out winners are not regulation scoreline wins. The outcome target is derived from
  the recorded full-time score columns, consistently with the official regression task.

## Point-in-time information sets

`pre_match` uses completed earlier matches/editions, known stage/format, strictly prior dynamic
state, rest, and aggregate player/coach history. `kickoff` may additionally use current roster,
starter, and coach identity while retaining only their prior events. `half_time` may use the
half-time score and predicts a nonnegative second-half increment. Current-match goals/cards are
applied to player histories only after the prediction snapshot.

Every generated feature has an availability class, source, aggregation cutoff, and schema hash.
Automated mutation tests verify that changing current-match event tokens cannot change that
match's `pre_match` or `kickoff` features.

## Historical coverage and dependence

The data contain only 20 tournament environments, with major changes in teams, matches, rules,
and competitive composition. Matches within an edition are dependent, later teams are selected
through earlier results, and editions are temporally ordered. Therefore:

- model selection uses expanding-origin tournament folds, never random match splits;
- uncertainty resamples tournament editions, not IID matches;
- 2014 is excluded from selection and used only for the frozen audit;
- post-2014 results and real 2026 outcomes are excluded from the closed track;
- 48-team/104-match output is an explicitly labelled format scenario.

## Known biases and limitations

Early tournaments have fewer teams and different qualification/geopolitical structures. Historical
team initials do not imply a causal continuity relation between states or federations. The data do
not contain bookmaker odds, injuries, substitutions as a complete state process, red-card timing
in a fully normalized form, contemporary rankings, or actual 2026 entrants/fixtures. Player event
tokens are sparse and not a comprehensive event feed. The sample cannot identify large neural
architectures reliably, and the 2014 audit is software-isolated but not researcher-blind because
earlier repository results were already visible.

## Governance and audit artifacts

The authoritative machine-readable evidence is written under `outputs/audit/`: raw hashes,
schema report, missingness, duplicate report, team-name registry, reconciliation, data lineage,
feature availability, and the final manifest. Any narrative discrepancy is resolved in favor of
those artifacts and the immutable protocol lock.
