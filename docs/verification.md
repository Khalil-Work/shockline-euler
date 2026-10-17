# Verification evidence

Reproduced by `python -m pytest tests -q` (12 tests) and
`python examples/run_verification.py` (figures + GIF + raw CSV/JSON under
`docs/results/`). Verified on Windows 11, Python 3.13.5, NumPy 2.1.3.

## What each test establishes

| Test | Property established |
| --- | --- |
| `test_primitive_conserved_round_trip` | State transforms are exact inverses (1e-14). |
| `test_rusanov_flux_matches_hand_computation` | The flux formula, including the ½ on the dissipation term, against an independent hand computation; identical states give the pure physical flux. This is the test that catches a doubled dissipation coefficient. |
| `test_uniform_state_is_preserved_exactly` | Freestream consistency for both schemes (1e-14). |
| `test_conservation_with_equal_boundary_fluxes` | With identical boundary fluxes (p uniform), mass/momentum/energy totals are conserved to 1e-13 for the second-order scheme. |
| `test_sod_momentum_balance_matches_boundary_flux_integral` | On Sod, the momentum change equals the exact pressure-force integral (p_L − p_R)·t = 0.18 to 1e-10 relative — conservation in the presence of pressure-supported open boundaries. |
| `test_positivity_throughout_sod_evolution` | ρ > 0 and p > 0 at every step of the first-order Sod run (the regime with a positivity guarantee). |
| `test_exact_riemann_star_state_matches_published_sod_values` | p* = 0.30313, u* = 0.92745 vs the published values (Toro §4.3.3). |
| `test_exact_riemann_rarefaction_fan_matches_analytic_profile` | Inside the fan, u, ρ, p match the exact self-similar isentropic formulas to 1e-12; far-field states exact. |
| `test_convergence_against_exact_advected_contact` | On grids N=100,200,400, checks orders above 0.8 (first) and 1.6 (second) against the exact transported contact; requires at least 3× lower second-order error at N=400. The separate recorded benchmark adds N=800. |
| `test_mc_limiter_introduces_no_new_extrema` | Reconstructed interface states stay within the data range on a discontinuous profile. |
| `test_invalid_inputs_raise_meaningful_errors` | gamma/t_end validation, non-finite state detection, and Riemann non-convergence signaling. |
| `test_vacuum_generating_expansion_is_rejected` | Positive primitive states moving apart fast enough to form a vacuum raise an explicit unsupported-domain error. |

## Recorded run (`docs/results/`)

- `sod_comparison.png` + `sod_history.csv`: Sod at t = 0.2, N = 400 vs the
  exact sampled Riemann solution (all quantities nondimensional). L1(ρ):
  first order 1.10e-2, MUSCL-SSP3 1.85e-3. This is a **single-grid accuracy
  comparison**: under refinement these L1 errors do tend to zero at a
  reduced rate (consistent shock-capturing convergence with discontinuities;
  the L∞ error at a moving jump need not vanish), which a fixed-grid table
  does not measure. The clean order evidence is the smooth-contact study.
- `contact_convergence.png` + `contact_convergence.csv`: exact-reference
  convergence through N = 800, orders →1 and →2 with reference slopes.
  Test-suite assertions cover N ≤ 400; the recorded benchmark extends the
  same trend to N = 800.
- `sod_evolution.gif`: 41 frames of the computed density against the exact
  profile, recorded from the second-order run.
- `run_summary.json`: metrics above + machine/runtime context.

## Limits of this evidence

- The exact-Riemann comparison validates wave physics for one standard
  shock-tube family (γ = 1.4, moderate shocks); strong shocks, vacuum
  limiters, and other γ values are not covered.
- Zero-gradient boundaries are not non-reflecting; every benchmark keeps
  waves inside the domain. Do not extend results beyond those times.
- Positivity is guaranteed only for the first-order scheme; the
  second-order scheme is positivity-monitored with stage checks and a CFL
  of 0.40, which has been sufficient for the cases tested here.
- No multi-dimensional, viscous, reacting, or source-term capability.
