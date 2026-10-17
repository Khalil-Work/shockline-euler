"""Verification tests for finite_volume (1D Euler finite-volume solvers).

Each test pins a distinct property:
- primitive/conserved round trip;
- the Rusanov flux formula against a hand-computed value (this is what
  catches a doubled dissipation coefficient);
- uniform states are preserved exactly (freestream consistency);
- total mass, momentum, and energy are conserved to roundoff on the Sod
  problem (boundary fluxes cancel by symmetry of the zero-gradient rule);
- density and pressure stay positive throughout the first-order Sod run;
- the exact Riemann solver reproduces the published Sod star state and
  recovers the far-field states exactly;
- first- and second-order convergence against the EXACT advected contact
  profile (rho, u=const, p=const is an exact Euler solution transported at
  speed u; the error magnitude is a direct measure of interface diffusion);
- the MC limiter introduces no new extrema on discontinuous data;
- invalid inputs and non-physical states raise meaningful errors.
"""

import numpy as np
import pytest

from finite_volume import (
    conserved,
    integrate,
    primitive,
    riemann_exact,
    rusanov_flux,
    sample_riemann,
    sod_initial,
    sound_speed,
    step_muscl_ssp3,
    step_rusanov,
)

GAMMA = 1.4


def sod_grid(n):
    x = (np.arange(n) + 0.5) / n
    rho, u, p = sod_initial(x)
    return x, conserved(rho, u, p, GAMMA)


def test_primitive_conserved_round_trip():
    rng = np.random.default_rng(3)
    rho = rng.uniform(0.5, 2.0, 50)
    u = rng.uniform(-2.0, 2.0, 50)
    p = rng.uniform(0.5, 3.0, 50)
    r2, u2, p2 = primitive(conserved(rho, u, p, GAMMA), GAMMA)
    np.testing.assert_allclose(r2, rho, rtol=1e-14)
    np.testing.assert_allclose(u2, u, rtol=1e-14)
    np.testing.assert_allclose(p2, p, rtol=1e-14)


def test_rusanov_flux_matches_hand_computation():
    # Left state rho=1, u=0.3, p=1; right state rho=0.5, u=-0.2, p=0.8.
    # cL = sqrt(1.4), cR = sqrt(1.4*0.8/0.5); smax = max(|uL|+cL, |uR|+cR).
    UL = conserved(1.0, 0.3, 1.0, GAMMA)
    UR = conserved(0.5, -0.2, 0.8, GAMMA)
    cL = np.sqrt(1.4 * 1.0 / 1.0)
    cR = np.sqrt(1.4 * 0.8 / 0.5)
    smax = max(abs(0.3) + cL, abs(-0.2) + cR)
    EL = 1.0 / 0.4 + 0.5 * 1.0 * 0.09
    ER = 0.8 / 0.4 + 0.5 * 0.5 * 0.04
    FL = np.array([1.0 * 0.3, 1.0 * 0.09 + 1.0, 0.3 * (EL + 1.0)])
    FR = np.array([0.5 * -0.2, 0.5 * 0.04 + 0.8, -0.2 * (ER + 0.8)])
    expected = 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)
    np.testing.assert_allclose(rusanov_flux(UL, UR, GAMMA), expected, rtol=1e-14)

    # Identical states: dissipation term must vanish exactly.
    F_same = rusanov_flux(UL, UL, GAMMA)
    np.testing.assert_allclose(F_same, FL, rtol=1e-14)


def test_uniform_state_is_preserved_exactly():
    U = conserved(np.full(40, 0.8), np.full(40, 0.3), np.full(40, 1.7), GAMMA)
    for order in (1, 2):
        U_end = integrate(U.copy(), t_end=0.3, dx=1.0 / 40, gamma=GAMMA, order=order)
        np.testing.assert_allclose(U_end, U, rtol=0, atol=1e-14)


def test_conservation_with_equal_boundary_fluxes():
    # The advected contact below decays to identical states at both ends
    # (rho = 1, u = 0.5, p = 1), so the zero-gradient boundary rule gives
    # identical boundary fluxes and every conserved total (mass, momentum,
    # energy) is preserved to roundoff.
    _, U0 = _advected_contact_initial(300)
    U1 = integrate(U0.copy(), t_end=0.1, dx=1.0 / 300, gamma=GAMMA, order=2)
    np.testing.assert_allclose(U1.sum(axis=1), U0.sum(axis=1), rtol=1e-13)


def test_sod_momentum_balance_matches_boundary_flux_integral():
    # For the Sod problem the waves never reach the boundaries, so the
    # boundary states keep their initial pressures and the total momentum
    # change must equal the exact pressure-force integral
    # (p_L - p_R) * t = (1.0 - 0.1) * 0.2 = 0.18.
    n = 400
    dx = 1.0 / n
    _, U0 = sod_grid(n)
    U1 = integrate(U0.copy(), t_end=0.2, dx=dx, gamma=GAMMA, order=2)
    delta = float((U1[1].sum() - U0[1].sum()) * dx)
    assert delta == pytest.approx(0.18, rel=1e-10)


def test_positivity_throughout_sod_evolution():
    U = sod_grid(400)[1]
    dx = 1.0 / 400
    t = 0.0
    while t < 0.2 - 1e-14:
        U, dt = step_rusanov(U, 0.2 - t, dx, GAMMA)
        t += dt
        rho, u, p = primitive(U, GAMMA)
        assert (rho > 0).all()
        assert (p > 0).all()


def test_exact_riemann_star_state_matches_published_sod_values():
    WL = (1.0, 0.0, 1.0)
    WR = (0.125, 0.0, 0.1)
    p_star, u_star = riemann_exact(WL, WR, GAMMA)
    # Toro (2009), section 4.3.3 / classic Sod reference values.
    assert p_star == pytest.approx(0.30313, abs=5e-5)
    assert u_star == pytest.approx(0.92745, abs=5e-5)


def test_exact_riemann_rarefaction_fan_matches_analytic_profile():
    WL = (1.0, 0.0, 1.0)
    WR = (0.125, 0.0, 0.1)
    np.testing.assert_allclose(sample_riemann(WL, WR, GAMMA, -3.0), (1.0, 0.0, 1.0), rtol=1e-12)
    np.testing.assert_allclose(sample_riemann(WL, WR, GAMMA, 3.0), (0.125, 0.0, 0.1), rtol=1e-12)
    # Inside the left rarefaction fan the exact self-similar solution is
    # u = 2/(gamma+1) * (cL + (gamma-1)/2 * uL + xi), rho/p isentropic.
    cL = np.sqrt(GAMMA * WL[2] / WL[0])
    pS, uS = riemann_exact(WL, WR, GAMMA)
    cSL = cL * (pS / WL[2]) ** ((GAMMA - 1.0) / (2.0 * GAMMA))
    xi = np.linspace(uL_fan(WL) + 0.02, uS - cSL - 0.02, 50)
    for s in xi:
        rho, u, p = sample_riemann(WL, WR, GAMMA, s)
        u_exact = 2.0 / (GAMMA + 1.0) * (cL + (GAMMA - 1.0) / 2.0 * WL[1] + s)
        c = u_exact - s
        rho_exact = WL[0] * (c / cL) ** (2.0 / (GAMMA - 1.0))
        p_exact = WL[2] * (c / cL) ** (2.0 * GAMMA / (GAMMA - 1.0))
        assert u == pytest.approx(u_exact, abs=1e-12)
        assert rho == pytest.approx(rho_exact, rel=1e-12)
        assert p == pytest.approx(p_exact, rel=1e-12)


def uL_fan(WL):
    return -np.sqrt(GAMMA * WL[2] / WL[0])


def _advected_contact_initial(n, amplitude=0.05, u0=0.5):
    # rho(x,0) = 1 + a*exp(-((x-0.5)/0.08)^2), u = u0, p = 1 is an exact
    # Euler solution: a passive contact/entropy wave transported at speed
    # u0, with rho(x,t) = rho(x-u0*t,0). The bump region stays away from
    # the boundaries for the integration times used here.
    x = (np.arange(n) + 0.5) / n
    rho = 1.0 + amplitude * np.exp(-((x - 0.5) / 0.08) ** 2)
    return x, conserved(rho, np.full(n, u0), np.full(n, 1.0), GAMMA)


def test_convergence_against_exact_advected_contact():
    t_end = 0.1  # profile travels to [0.35, 0.65]; far from boundaries
    errors = {1: [], 2: []}
    for order in (1, 2):
        for n in (100, 200, 400, 800):
            x, U0 = _advected_contact_initial(n)
            rho = primitive(integrate(U0, t_end, 1.0 / n, GAMMA, order=order), GAMMA)[0]
            exact = 1.0 + 0.05 * np.exp(-((x - 0.5 - 0.5 * t_end) / 0.08) ** 2)
            errors[order].append(float(np.mean(np.abs(rho - exact))))

    def order_of(errs):
        return np.log2(np.array(errs[:-1]) / np.array(errs[1:]))

    assert np.all(order_of(errors[1]) > 0.8), f"first order: {errors[1]}"
    assert np.all(order_of(errors[2]) > 1.6), f"second order: {errors[2]}"
    # The second-order scheme must resolve the contact far better than the
    # first-order one at the finest resolution (diffusion-sensitive check).
    assert errors[2][-1] < errors[1][-1] / 3.0


def test_mc_limiter_introduces_no_new_extrema():
    from finite_volume.euler1d import _mc_limited_slope

    q = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 2.0, 0.0, 0.0])
    slope = _mc_limited_slope(q)
    left = q[1:-1] + 0.5 * slope[1:-1]
    right = q[1:-1] - 0.5 * slope[1:-1]
    lo, hi = q.min(), q.max()
    assert (left >= lo - 1e-15).all() and (left <= hi + 1e-15).all()
    assert (right >= lo - 1e-15).all() and (right <= hi + 1e-15).all()


def test_invalid_inputs_raise_meaningful_errors():
    U = conserved(np.full(10, 1.0), np.zeros(10), np.full(10, 1.0), GAMMA)
    with pytest.raises(ValueError, match="gamma"):
        integrate(U.copy(), 0.1, 0.1, gamma=1.0)
    with pytest.raises(ValueError, match="gamma"):
        integrate(U.copy(), 0.1, 0.1, gamma=float("nan"))
    with pytest.raises(ValueError, match="t_end"):
        integrate(U.copy(), 0.0, 0.1, gamma=GAMMA)
    with pytest.raises(ValueError, match="t_end"):
        integrate(U.copy(), float("nan"), 0.1, gamma=GAMMA)
    with pytest.raises(ValueError, match="t_end"):
        integrate(U.copy(), -0.1, 0.1, gamma=GAMMA)
    with pytest.raises(ValueError, match="dx"):
        integrate(U.copy(), 0.1, float("nan"), gamma=GAMMA)
    with pytest.raises(ValueError, match="order"):
        integrate(U.copy(), 0.1, 0.1, gamma=GAMMA, order=17)
    with pytest.raises(ValueError, match="dt_max"):
        step_rusanov(U.copy(), -0.01, 0.1, GAMMA)
    with pytest.raises(ValueError, match="cfl"):
        step_muscl_ssp3(U.copy(), 0.01, 0.1, GAMMA, cfl=float("nan"))
    broken = U.copy()
    broken[0, 3] = np.nan
    with pytest.raises(FloatingPointError):
        integrate(broken, 0.1, 0.1, gamma=GAMMA, order=1)
    with pytest.raises(ValueError, match="non-vacuum"):
        riemann_exact((0.0, 0.0, 1.0), (0.125, 0.0, 0.1), GAMMA)
    with pytest.raises(ValueError, match="finite"):
        riemann_exact((1.0, 0.0, float("nan")), (0.125, 0.0, 0.1), GAMMA)
    with pytest.raises(ValueError, match="gamma"):
        riemann_exact((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), float("nan"))
    with pytest.raises(RuntimeError, match="converge"):
        riemann_exact((1.0, 0.0, 1.0), (1.0, 0.0, 1.0e20), GAMMA, maxit=5)
