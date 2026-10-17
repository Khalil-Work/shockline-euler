"""One-dimensional compressible Euler equations: finite-volume solvers and an
exact Riemann solver for verification.

Conserved variables U = (rho, rho*u, E) with total energy
E = p/(gamma-1) + rho*u^2/2 on a uniform grid. Two schemes are provided:

- ``step_rusanov``: first-order conservative update with the Rusanov
  (local Lax-Friedrichs) approximate Riemann flux
  F = (F(UL)+F(UR))/2 - smax*(UR-UL)/2, which is positivity preserving
  under the CFL condition dt < dx / max(|u| + c).
- ``step_muscl_ssp3``: second-order MUSCL reconstruction of primitive
  variables with the MC (monotonized-central) limiter, the same Rusanov
  flux at the interface states, and three-stage SSP Runge-Kutta time
  stepping. The limiter prevents new extrema at the reconstructed
  interfaces; positivity of the second-order scheme is monitored and
  enforced by CFL choice (0.4 by default), not guaranteed unconditionally.

``riemann_exact`` implements the exact iterative Riemann solver for a
gamma-law gas (Newton iteration on the star pressure, Toro ch. 4), used as
the verification reference for shock-tube problems.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "conserved",
    "primitive",
    "sound_speed",
    "rusanov_flux",
    "step_rusanov",
    "step_muscl_ssp3",
    "integrate",
    "riemann_exact",
    "sample_riemann",
    "sod_initial",
]


def conserved(rho, u, p, gamma):
    """Primitive (rho, u, p) -> conserved (rho, rho*u, E)."""
    energy = p / (gamma - 1.0) + 0.5 * rho * u * u
    return np.stack([rho, rho * u, energy])


def primitive(U, gamma):
    """Conserved -> primitive, with sound speed c."""
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u * u)
    return rho, u, p


def sound_speed(rho, p, gamma):
    return np.sqrt(gamma * p / rho)


def rusanov_flux(UL, UR, gamma):
    """Rusanov (local Lax-Friedrichs) numerical flux at one interface.

    UL, UR are conserved states (shape (3,)); returns the flux (3,).
    """
    rhoL, uL, pL = primitive(UL, gamma)
    rhoR, uR, pR = primitive(UR, gamma)
    cL = sound_speed(rhoL, pL, gamma)
    cR = sound_speed(rhoR, pR, gamma)
    smax = max(abs(uL) + cL, abs(uR) + cR)

    def physical_flux(rho, u, p):
        return np.array(
            [rho * u, rho * u * u + p, u * (p / (gamma - 1.0) + 0.5 * rho * u * u + p)]
        )

    return 0.5 * (physical_flux(rhoL, uL, pL) + physical_flux(rhoR, uR, pR)) - 0.5 * smax * (
        UR - UL
    )


def _interfaces_first_order(U, gamma):
    """Left/right states at interfaces for a piecewise-constant scheme."""
    n_cells = U.shape[1]
    UL = U[:, :-1]
    UR = U[:, 1:]
    return UL, UR


def _mc_limited_slope(q):
    """Monotonized-central (MC) limited slope of a scalar field.

    minmod(2*dqm, 2*dqp, dq0) with zero slopes at the ends; prevents
    new extrema in the reconstructed interface states and returns the
    central difference dq0 on smooth monotone data.
    """
    dqm = q[1:-1] - q[:-2]
    dqp = q[2:] - q[1:-1]
    dq0 = 0.5 * (q[2:] - q[:-2])
    slope = np.where(
        (dqm * dqp > 0),
        np.sign(dq0) * np.minimum(2.0 * np.abs(dqm), np.minimum(2.0 * np.abs(dqp), np.abs(dq0))),
        0.0,
    )
    return np.concatenate([[0.0], slope, [0.0]])


def _interface_states(U, gamma):
    """MUSCL interface states with the MC limiter (zero ghost slopes).

    Reconstruction operates on primitive variables (rho, u, p): limited
    conserved-variable slopes can combine into non-physical interface
    pressures at contacts, while primitive reconstruction keeps interface
    states meaningful. Interface i uses left state w_i + slope_i/2 and
    right state w_{i+1} - slope_{i+1}/2.
    """
    rho, u, p = primitive(U, gamma)
    w = np.stack([rho, u, p])
    slopes = np.stack([_mc_limited_slope(w[k]) for k in range(3)])
    wL = w[:, :-1] + 0.5 * slopes[:, :-1]
    wR = w[:, 1:] - 0.5 * slopes[:, 1:]
    rL, uL, pL = wL
    rR, uR, pR = wR
    UL = np.stack([rL, rL * uL, pL / (gamma - 1.0) + 0.5 * rL * uL * uL])
    UR = np.stack([rR, rR * uR, pR / (gamma - 1.0) + 0.5 * rR * uR * uR])
    return UL, UR


def _flux_divergence(U, gamma, dx, limiter):
    n_cells = U.shape[1]
    if limiter:
        UL, UR = _interface_states(U, gamma)
    else:
        UL, UR = _interfaces_first_order(U, gamma)
    flux = np.empty((3, UL.shape[1]))
    for i in range(UL.shape[1]):
        flux[:, i] = rusanov_flux(UL[:, i], UR[:, i], gamma)
    # Flux at the two boundary interfaces uses the boundary states
    # themselves (zero-gradient ghost cells): waves leave, none enter.
    left_ghost = U[:, :1]
    right_ghost = U[:, -1:]
    flux_left = rusanov_flux(left_ghost[:, 0], U[:, 0], gamma).reshape(3, 1)
    flux_right = rusanov_flux(U[:, -1], right_ghost[:, 0], gamma).reshape(3, 1)
    flux = np.concatenate([flux_left, flux, flux_right], axis=1)
    return -(flux[:, 1:] - flux[:, :-1]) / dx


def _cfl_dt(U, dx, gamma, cfl):
    rho, u, p = primitive(U, gamma)
    c = sound_speed(rho, p, gamma)
    return cfl * dx / np.max(np.abs(u) + c)


def _check_state(U, gamma, where):
    """Reject non-finite or non-physical states with a specific error."""
    if not np.all(np.isfinite(U)):
        raise FloatingPointError(
            f"{where}: conserved state contains non-finite values; the "
            "scheme lost stability (reduce dt/CFL or check the data)"
        )
    rho, u, p = primitive(U, gamma)
    if np.any(rho <= 0.0) or np.any(p <= 0.0):
        i = int(np.argmin(np.minimum(rho, p)))
        raise FloatingPointError(
            f"{where}: non-physical state at cell {i} "
            f"(rho={rho[i]:.3e}, p={p[i]:.3e}); reduce the CFL number"
        )


def _check_positive_finite(value, name):
    """Require a finite, strictly positive scalar (NaN fails both tests)."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


def _check_gamma(gamma):
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError(f"gamma must be finite and > 1, got {gamma!r}")


def step_rusanov(U, dt_max, dx, gamma, cfl=0.45):
    """One first-order Rusanov step; dt is capped by ``dt_max`` and the CFL."""
    _check_gamma(gamma)
    _check_positive_finite(dx, "dx")
    _check_positive_finite(dt_max, "dt_max")
    _check_positive_finite(cfl, "cfl")
    dt = min(dt_max, _cfl_dt(U, dx, gamma, cfl))
    result = U + dt * _flux_divergence(U, gamma, dx, limiter=False)
    _check_state(result, gamma, "step_rusanov")
    return result, dt


def step_muscl_ssp3(U, dt_max, dx, gamma, cfl=0.40):
    """One second-order MUSCL-Rusanov step with SSP-RK3 time stepping.

    Stages: U1 = U + dt L(U); U2 = 3/4 U + 1/4 (U1 + dt L(U1));
    result = 1/3 U + 2/3 (U2 + dt L(U2)). The three-stage
    strong-stability-preserving Runge-Kutta method is used because the
    limited semi-discrete operator is nonlinear; dt is capped by ``dt_max``
    and the CFL condition. Each stage state is checked for validity.
    """
    _check_gamma(gamma)
    _check_positive_finite(dx, "dx")
    _check_positive_finite(dt_max, "dt_max")
    _check_positive_finite(cfl, "cfl")
    dt = min(dt_max, _cfl_dt(U, dx, gamma, cfl))
    k1 = _flux_divergence(U, gamma, dx, limiter=True)
    U1 = U + dt * k1
    _check_state(U1, gamma, "step_muscl_ssp3 stage 1")
    k2 = _flux_divergence(U1, gamma, dx, limiter=True)
    U2 = 0.75 * U + 0.25 * (U1 + dt * k2)
    _check_state(U2, gamma, "step_muscl_ssp3 stage 2")
    k3 = _flux_divergence(U2, gamma, dx, limiter=True)
    result = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * k3)
    _check_state(result, gamma, "step_muscl_ssp3")
    return result, dt


def integrate(U0, t_end, dx, gamma, order=2):
    """Integrate to exactly ``t_end`` (last step trimmed); returns U.

    Requires a finite positive ``t_end``, positive spacing, a finite gamma
    above one, and ``order`` exactly 1 or 2. Raises RuntimeError if a step
    makes no time progress (degenerate CFL).
    """
    _check_positive_finite(t_end, "t_end")
    _check_gamma(gamma)
    _check_positive_finite(dx, "dx")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    step = step_muscl_ssp3 if order == 2 else step_rusanov
    U = U0.copy()
    t = 0.0
    while t < t_end - 1e-14:
        U, dt = step(U, t_end - t, dx, gamma)
        if dt <= 0.0 or not np.isfinite(dt):
            raise RuntimeError(f"no time progress at t={t:.6e} (dt={dt})")
        t += dt
    return U


# ----------------------------------------------------------------------
# Exact Riemann solver (Toro, ch. 4) for a gamma-law gas
# ----------------------------------------------------------------------


def _f(p, rho, u, pK, gamma):
    """Wave-curve function for one side; K in {L, R} implied by arguments."""
    c = sound_speed(rho, pK, gamma)
    if p > pK:  # shock
        A = 2.0 / (rho * ((gamma + 1.0) * p + (gamma - 1.0) * pK))
        return (p - pK) * np.sqrt(A)
    B = pow(p / pK, (gamma - 1.0) / (2.0 * gamma))
    return 2.0 * c / (gamma - 1.0) * (B - 1.0)


def _f_prime(p, rho, pK, gamma):
    c = sound_speed(rho, pK, gamma)
    if p > pK:
        A = 2.0 / (rho * ((gamma + 1.0) * p + (gamma - 1.0) * pK))
        return np.sqrt(A) * (1.0 - 0.5 * (p - pK) / (p + (gamma - 1.0)
                                                      / (gamma + 1.0) * pK))
    return c / (gamma * pK) * pow(p / pK, -(gamma + 1.0) / (2.0 * gamma))


def riemann_exact(WL, WR, gamma, tol=1e-12, maxit=50):
    """Exact star-state solution of the Riemann problem.

    WL, WR: (rho, u, p) tuples for the left/right states. Returns
    (p_star, u_star). Supported domain: positive, finite, non-vacuum states
    and gamma > 1; anything else raises ``ValueError`` (the solver does not
    handle vacuum or negative-temperature inputs).
    """
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    _check_gamma(gamma)
    states = (rhoL, uL, pL, rhoR, uR, pR)
    if not all(np.isfinite(states)):
        raise ValueError(f"Riemann input states must be finite, got {states}")
    if rhoL <= 0.0 or rhoR <= 0.0 or pL <= 0.0 or pR <= 0.0:
        raise ValueError(
            f"Riemann solver supports positive non-vacuum states only, got "
            f"rhoL={rhoL}, pL={pL}, rhoR={rhoR}, pR={pR}"
        )
    # Two-rarefaction initial guess (Toro eq. 4.46).
    cL = sound_speed(rhoL, pL, gamma)
    cR = sound_speed(rhoR, pR, gamma)
    num = cL + cR - 0.5 * (gamma - 1.0) * (uR - uL)
    power = (gamma - 1.0) / (2.0 * gamma)
    denom_base = cL * pow(pL / np.sqrt(pL * pR), -power) + cR * pow(
        pR / np.sqrt(pL * pR), -power
    )
    p = (num / denom_base) ** (1.0 / power)
    p = max(p, 1e-8 * min(pL, pR))
    for it in range(maxit):
        f = _f(p, rhoL, uL, pL, gamma) + _f(p, rhoR, uR, pR, gamma) + (uR - uL)
        dp = f / (_f_prime(p, rhoL, pL, gamma) + _f_prime(p, rhoR, pR, gamma))
        p -= dp
        if p <= 0.0:
            p = 1e-8 * min(pL, pR)
            continue
        if abs(dp) < tol * p:
            break
    else:
        raise RuntimeError(
            f"exact Riemann Newton iteration did not converge in {maxit} "
            f"iterations (p*={p:.6e}); check the input states"
        )
    u = 0.5 * (uL + uR) + 0.5 * (_f(p, rhoR, uR, pR, gamma) - _f(p, rhoL, uL, pL, gamma))
    return p, u


def sample_riemann(WL, WR, gamma, xi):
    """Sample the exact Riemann solution at wave speeds xi = x / t."""
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    pS, uS = riemann_exact(WL, WR, gamma)
    cL = sound_speed(rhoL, pL, gamma)
    cR = sound_speed(rhoR, pR, gamma)

    if xi < uS:
        if pS > pL:  # left shock
            SL = uL - cL * np.sqrt(
                (gamma + 1.0) / (2.0 * gamma) * pS / pL + (gamma - 1.0) / (2.0 * gamma)
            )
            if xi < SL:
                return rhoL, uL, pL
            rho = rhoL * (
                (pS / pL + (gamma - 1.0) / (gamma + 1.0))
                / ((gamma - 1.0) / (gamma + 1.0) * pS / pL + 1.0)
            )
            return rho, uS, pS
        # left rarefaction
        cSL = cL * pow(pS / pL, (gamma - 1.0) / (2.0 * gamma))
        if xi < uL - cL:
            return rhoL, uL, pL
        if xi > uS - cSL:
            return rhoL * pow(pS / pL, 1.0 / gamma), uS, pS
        u = 2.0 / (gamma + 1.0) * (cL + (gamma - 1.0) / 2.0 * uL + xi)
        c = u - xi
        rho = rhoL * pow(c / cL, 2.0 / (gamma - 1.0))
        p = pL * pow(c / cL, 2.0 * gamma / (gamma - 1.0))
        return rho, u, p
    if pS > pR:  # right shock
        SR = uR + cR * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * pS / pR + (gamma - 1.0) / (2.0 * gamma)
        )
        if xi > SR:
            return rhoR, uR, pR
        rho = rhoR * (
            (pS / pR + (gamma - 1.0) / (gamma + 1.0))
            / ((gamma - 1.0) / (gamma + 1.0) * pS / pR + 1.0)
        )
        return rho, uS, pS
    # right rarefaction
    cSR = cR * pow(pS / pR, (gamma - 1.0) / (2.0 * gamma))
    if xi > uR + cR:
        return rhoR, uR, pR
    if xi < uS + cSR:
        return rhoR * pow(pS / pR, 1.0 / gamma), uS, pS
    u = 2.0 / (gamma + 1.0) * (-cR + (gamma - 1.0) / 2.0 * uR + xi)
    c = xi - u
    rho = rhoR * pow(c / cR, 2.0 / (gamma - 1.0))
    p = pR * pow(c / cR, 2.0 * gamma / (gamma - 1.0))
    return rho, u, p


def sod_initial(x, x0=0.5):
    """Standard Sod shock-tube initial states on grid centers ``x``."""
    left = x < x0
    rho = np.where(left, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(left, 1.0, 0.1)
    return rho, u, p
