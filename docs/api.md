# API reference

All functions live in `finite_volume.euler1d` (re-exported from
`finite_volume`). Arrays are shape `(3, n_cells)` conserved states unless
stated. Units: SI with the domain length in meters.

## State transforms

- `conserved(rho, u, p, gamma) -> (3, n)` — primitives to conserved.
- `primitive(U, gamma) -> (rho, u, p)` — conserved to primitives.
- `sound_speed(rho, p, gamma) -> c` — elementwise c = sqrt(gamma p / rho).

## Fluxes and steps

- `rusanov_flux(UL, UR, gamma) -> (3,)` — interface flux
  ½(F_L+F_R) − ½ s_max (U_R−U_L), s_max the larger signal speed.
- `step_rusanov(U, dt_max, dx, gamma, cfl=0.45) -> (U_new, dt)` — one
  first-order step; dt = min(dt_max, CFL-limited). Raises `ValueError` for
  gamma ≤ 1 or dx ≤ 0 and `FloatingPointError` if the state becomes
  non-finite or non-physical.
- `step_muscl_ssp3(U, dt_max, dx, gamma, cfl=0.40) -> (U_new, dt)` — one
  second-order MUSCL/SSP-RK3 step; every stage state is checked.

## Integration

- `integrate(U0, t_end, dx, gamma, order=2) -> U` — runs to exactly
  `t_end` (last step trimmed). `order=1` selects Rusanov, `order=2`
  MUSCL-SSP3. Raises `ValueError` for non-positive `t_end` and
  `RuntimeError` on zero time progress.

## Exact Riemann reference

- `riemann_exact(WL, WR, gamma, tol=1e-12, maxit=50) -> (p_star, u_star)` —
  Newton iteration on the star pressure; `RuntimeError` if not converged.
- `sample_riemann(WL, WR, gamma, xi) -> (rho, u, p)` — exact solution at
  the self-similar coordinate ξ = (x − x₀)/t. **ξ must be a scalar** (the
  wave-branch logic is scalar); vectorize with a loop over points.
- `sod_initial(x, x0=0.5)` — standard Sod left/right primitives.

## Example

```python
import numpy as np
from finite_volume import conserved, integrate, primitive, sod_initial

n, dx = 400, 1.0 / 400
x = (np.arange(n) + 0.5) / n
U0 = conserved(*sod_initial(x), gamma=1.4)
rho, u, p = primitive(integrate(U0, t_end=0.2, dx=dx, gamma=1.4, order=2), 1.4)
```
