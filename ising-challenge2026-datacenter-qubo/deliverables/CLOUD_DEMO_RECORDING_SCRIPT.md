# Cloud Demonstration Recording Script

Target duration: **2–3 minutes**. Record a genuine platform workflow; do not construct a mock or edited result. Hide account identifiers, API credentials, browser autofill, balances, notifications, and unrelated tasks.

## Before recording

- Use a clean browser/profile and disable desktop notifications.
- Prepare the exact 700-spin `peak_shave_lns` QUBO matrix and its index/metadata files from the frozen evidence package.
- Keep the independent replay report and final metrics ready in a separate window.
- Set the screen to a readable resolution and enlarge text where necessary.
- Never expose the plaintext credential file or paste a credential on screen.

## Shot list and narration

### 0:00–0:15 — Problem and scope

Show the project title and one sentence of scope:

> TAILOR-DC-SAFE models data-center action selection as bounded QUBO/Ising neighborhoods and accepts a candidate only after independent full-schedule business replay.

State explicitly that the cloud task covers a 700-spin neighborhood, not the entire 1,929-variable scheduling problem and not a global-optimality proof.

### 0:15–0:40 — Model artifact

Show the QUBO matrix dimensions, binary problem type, `peak_shave_lns` family name, and the corresponding variable-index file. Briefly explain that each spin maps to a candidate action in the bounded neighborhood.

### 0:40–1:15 — Platform call

On the real “玉盘·伊辛云” interface or SDK terminal:

- select/upload the same QUBO matrix;
- show `Binary/QUBO` and the 700-variable size;
- select the actual cloud solver;
- submit or reopen the genuine completed task;
- show the platform task status and returned energy/assignment without exposing credentials.

The current frozen package preserves the historical task ID, returned length,
and energy summary, but **does not preserve the raw 700-bit vector**. If the
historical task can still be opened, say this explicitly before retrieving it.
If it cannot be reopened, run a new genuine same-QUBO demonstration and label it
as a fresh workflow demonstration, not as the source of the frozen final result.

### 1:15–1:45 — Retrieval and energy check

If the platform exposes the historical assignment, retrieve and save the actual
700-bit vector during the recording, verify its length, and compute its energy
against the exact displayed QUBO. Rebuild the evidence package and paper before
submission if this closes the current provenance gap. If the assignment is not
available, do **not** claim that it was previously persisted and do not fabricate
an energy reconstruction; instead show a fresh same-QUBO result and state that it
demonstrates the cloud call path only.

### 1:45–2:20 — Merge and independent replay

Show the candidate being merged into the 588-job schedule and the independent `decode → repair → guard → replay` result. Display these verified final metrics:

- 588 selected jobs;
- 1,929 action variables/spins in the full model;
- 0 one-hot violations;
- 0 dummy actions;
- action cost: 5,414.858289 USD;
- expected recourse cost: 35.39452387390732 USD;
- CVaR95 recourse cost: 47.718284903473034 USD;
- peak IT load: 0.031646304 MW.

Explain that promotion required strict improvement over the predecessor in expected recourse, CVaR95, and peak IT load while preserving feasibility and the action-cost guard.

### 2:20–2:45 — Reproducibility and limitations

Show the strict verification report with `READY`, the clean-room validator result, and the archive checksum. End with the bounded claim:

> The frozen evidence confirms one successful cloud task summary and an independently replayable post-polish full schedule. This recording demonstrates the genuine cloud workflow. Unless the historical raw vector is recovered and rebound, it does not prove bit-level replay, isolate the cloud seed's causal contribution, establish global optimality, or estimate repeated-run variance.

## Final video QA

- [ ] MP4 opens from beginning to end with audible narration and readable text.
- [ ] No credential, account identifier, balance, phone number, email, or unrelated task is visible.
- [ ] The platform task, 700-spin scope, result retrieval, merge, and replay are shown; an assignment/energy check is claimed only when the genuine raw vector is actually available.
- [ ] The narration distinguishes the historical frozen result from any fresh demonstration run.
- [ ] Claims match the paper and reports; no global-optimality or full-cloud-solve language appears.
- [ ] The final attachment builder accepts the MP4 with `--require-video`.
