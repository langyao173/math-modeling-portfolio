# V14.1 Promote Decision

Decision: do not promote; keep opt-in

## Relabel-Only Gate

- Mean Final delta: 0.003027
- Mean B002 success delta: 0.024219
- Synthetic affine TrackB delta: 0.000000
- Invalid count OK: True
- p95 compile OK: True

## Public Means

- v13_root: Final=0.802881, TrackA=0.867383, TrackB=0.609375, B002=0.583984, invalid=0, max_p95_compile=3.582s
- v13_plus_b_relabel_only: Final=0.805908, TrackA=0.867383, TrackB=0.621484, B002=0.608203, invalid=0, max_p95_compile=3.998s
- v13_plus_tracka_exact_router_only: Final=0.805225, TrackA=0.870508, TrackB=0.609375, B002=0.583984, invalid=0, max_p95_compile=9.127s
- v13_plus_tracka_templates_only: Final=0.803467, TrackA=0.868164, TrackB=0.609375, B002=0.583984, invalid=0, max_p95_compile=5.907s
- v14_combined: Final=0.808838, TrackA=0.871289, TrackB=0.621484, B002=0.608203, invalid=0, max_p95_compile=10.631s

## Holdouts

- v13_root: affine_mean=0.606172, small_tracka_mean=0.843750
- v13_plus_b_relabel_only: affine_mean=0.606172, small_tracka_mean=0.843750
- v13_plus_tracka_exact_router_only: affine_mean=0.606172, small_tracka_mean=0.843750
- v13_plus_tracka_templates_only: affine_mean=0.606172, small_tracka_mean=0.843750
- v14_combined: affine_mean=0.606172, small_tracka_mean=0.843750
