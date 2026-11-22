# Method and Format Sources

## Supplied primary sources

- `A题 世界杯比赛数据挖掘与预测问题.pdf` — official task statement bundled with the project.
- `WorldCups_preview.csv`, `WorldCupMatches.csv`, `WorldCupPlayers.csv` — immutable official competition inputs bundled under `data/raw/`.

## Research-method sources

The complete 58-source screening matrix, inclusion logic, evidence classification, and nearest-work
comparison are in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md). The two entries below remain the
minimum methodological anchors rather than the complete bibliography.

- Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Applied Statistics, 46(2), 265–280. DOI: <https://doi.org/10.1111/1467-9876.00065>
- Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*. Journal of the American Statistical Association, 102(477), 359–378. DOI: <https://doi.org/10.1198/016214506000001437>

## 2026 structural scenario source

- FIFA, *How the FIFA World Cup 26 will work with 48 teams*: <https://www.fifa.com/en/articles/article-fifa-world-cup-2026-mexico-canada-usa-new-format-tournament-football-soccer>
- FIFA, *Regulations for the FIFA World Cup 2026*: <https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf>

The FIFA page supports the exogenous format totals; the official regulations bind the M73--M104
fixture graph and all 495 Annex C third-place assignments. No 2026 match result, live ranking,
roster performance, or post-kickoff fact is ingested into model fitting or validation.

## Source governance

- Primary papers, official documentation, and official format announcements are preferred.
- Post-2014 publications may motivate methods but cannot contribute match outcomes or fitted
  covariates to the closed track.
- Search-result snippets are discovery aids, not evidence. A source enters the matrix only after
  its paper/official page metadata and relevant claim are verified.
- No source supports a claim of causal effect, universal superiority, or guaranteed 2026 coverage.
- The novelty ledger distinguishes established components from the system-level integration tested
  here; absence of an identical prior system is not treated as proof of novelty.
