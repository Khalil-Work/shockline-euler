# Model and method

## Problem

One-dimensional compressible inviscid flow (Euler equations) on a uniform
grid of N finite volumes over x ∈ [0, 1]:

    ∂U/∂t + ∂F/∂x = 0,
    U = (ρ, ρu, E),   F = (ρu, ρu² + p, u(E + p)),
    E = p/(γ−1) + ρu²/2,

with density ρ, velocity u, pressure p, total energy E per unit volume, and
adiabatic index γ (1.4 for the benchmarks). Sound speed c = √(γp/ρ).
**Units:** the solver is unit-agnostic — it operates on whatever consistent
units the states carry. Every benchmark in this repository uses the standard
**nondimensional** Sod convention (states 1/0.125 and 1/0.1, domain [0, 1],
time to 0.2), so all figures, tables, and CSVs report nondimensional
quantities. One spatial dimension only — this is a 1D solver and is not
presented as anything else.

## Finite-volume update

Each cell stores the cell average of U. The semi-discrete form is

    dU_i/dt = −(F_{i+1/2} − F_{i−1/2}) / Δx,

and the scheme is conservative by construction (interior fluxes telescope).

### Rusanov (local Lax–Friedrichs) flux

    F̂_{i+1/2} = ½(F(U_L) + F(U_R)) − ½ s_max (U_R − U_L),
    s_max = max(|u_L| + c_L, |u_R| + c_R).

The dissipation coefficient carries the ½ factor; omitting it doubles the
numerical viscosity (a bug caught by the hand-computed flux test). First
order (piecewise constant) states with this flux are positivity preserving
under dt < Δx / max(|u| + c).

### MUSCL reconstruction with the MC limiter

Interface states are reconstructed from **primitive** variables
w = (ρ, u, p) with the monotonized-central slope

    σ_i = minmod(2Δw_{i−1/2}, 2Δw_{i+1/2}, (Δw_{i−1/2} + Δw_{i+1/2})/2),

    w_{i+1/2}^L = w_i + σ_i/2,   w_{i+1/2}^R = w_{i+1} − σ_{i+1}/2,

zero slopes at the domain ends. Primitive (not conserved) reconstruction
keeps interface pressures meaningful at contacts. The limiter prevents new
extrema at the reconstructed interfaces; it degenerates to first order at
smooth extrema, the standard TVD behavior. Positivity of the second-order
scheme is monitored (stage checks) and managed by the CFL number, not
guaranteed unconditionally.

### Time integration

First-order scheme: forward Euler with the CFL-limited step. Second-order
scheme: the three-stage strong-stability-preserving Runge–Kutta method

    U¹ = U + Δt L(U)
    U² = ¾ U + ¼ (U¹ + Δt L(U¹))
    Uⁿ⁺¹ = ⅓ U + ⅔ (U² + Δt L(U²))

with CFL defaults 0.45 (first order) and 0.40 (MUSCL); the last step is
trimmed to hit t_end exactly.

## Boundaries

Zero-gradient (outflow) ghost states: the boundary interface flux equals the
physical flux of the boundary cell. This is **not** a non-reflecting
boundary condition — subsonic incoming characteristics can be contaminated.
All benchmarks here are run for times at which the waves stay well inside
the domain; keep that restriction when adding new cases.

## Exact Riemann solver (verification reference)

The star state is found by Newton iteration on the standard pressure
function (Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
ch. 4): shocks use the exact Rankine–Hugoniot jump, rarefactions the
isentropic branch, with the two-rarefaction initial guess. Non-convergence
raises `RuntimeError`. The sampler `sample_riemann(WL, WR, γ, ξ)` evaluates
the exact self-similar solution at ξ = (x − x₀)/t, including both fan
interiors via the isentropic relations.
