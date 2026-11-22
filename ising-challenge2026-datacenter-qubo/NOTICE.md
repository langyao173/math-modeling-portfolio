# Notices and third-party materials

This file records provenance and redistribution boundaries. It is descriptive
and does not replace the terms supplied by a dataset, software vendor,
competition organizer, or applicable law.

## Project-level licensing status

No top-level license file was found for the original TAOLIN-DC-SIM source and
generated artifacts at the time of this audit. This repository therefore does
not assert a general open-source or open-data license for those materials. Do
not infer redistribution rights from their presence in a local research or
competition workspace.

Competition submission permission and public redistribution are separate
questions. Before publishing or redistributing a bundle, verify the competition
rules and every upstream source's current terms.

## Third-party datasets and repositories

The workspace contains or references material obtained from the following
sources:

- Alibaba Cluster Data GPU traces and repository material. Preserve the
  upstream notices included under
  [`data/raw/workload/alibaba_gpu`](data/raw/workload/alibaba_gpu/).
- Google Cluster Data repository material. See the included
  [upstream README](data/raw/workload/google_cluster/README.md).
- PGLib-OPF cases. Preserve its included
  [license file](data/raw/grid/pglib/LICENSE) and
  [README](data/raw/grid/pglib/README.md).
- NLR/NREL NSRDB and Wind Toolkit API samples. Access conditions, attribution,
  and redistribution requirements remain governed by the upstream services.
- CAISO OASIS is referenced by the downloader, but the canonical processed
  inputs do not contain successfully downloaded CAISO price/demand data; they
  use a synthetic fallback instead.

The presence of source data, transformed Parquet files, validation CSV files,
or generated figures does not change the upstream rights attached to the source
material. See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for the exact role of
each source in the canonical experiment.

## Plectrum SDK

The local competition materials include an organizer-provided
`plectrum_core-0.1.1-py3-none-any.whl`. The wheel is not part of this project
tree and is not declared as a public package dependency. No redistribution or
modification rights for that wheel are asserted here.

For private reproduction, obtain the SDK from the organizer or other authorized
source. Include it in a submission/public bundle only after confirming that the
applicable terms permit doing so. The frozen-result validator does not require
the wheel.

## Services and credentials

CloudSolver, NSRDB, Wind Toolkit, and other API-backed reruns may require service
accounts, API credentials, quotas, or credits. Credentials are not project
artifacts and must not be committed, logged, placed in documentation, or added
to a submission archive. Use process-scoped environment variables and follow
the service provider's terms.

## Generated results

The submitted metrics are results of the documented transformation and replay
pipeline. They are not warranties of market performance, global optimality, or
fitness for operational grid or data-center control. Synthetic inputs and model
limitations are listed in the project README and data-provenance record.
