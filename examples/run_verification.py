"""Produce all verification results, figures, animation, and raw data.

Runs from the repository root:

    python examples/run_verification.py

Outputs under ``docs/results/``:
- ``sod_comparison.png/.svg``       cover: Sod tube at t = 0.2 vs exact
- ``sod_evolution.gif``             animated density profiles vs exact
- ``contact_convergence.png/.svg``  error vs grid for both schemes
- ``sod_history.csv``, ``contact_convergence.csv``, ``run_summary.json``
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from finite_volume import (
    conserved,
    integrate,
    primitive,
    riemann_exact,
    sample_riemann,
    sod_initial,
)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREEN, GRAY = "#1f4e79", "#b23a48", "#0b6e4f", "#555555"
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)

GAMMA = 1.4  # all benchmark states below are nondimensional
WL, WR = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)
T_SOD = 0.2


def exact_profile(x, t):
    """Exact Sod solution on the cell centers; nondimensional variables.

    At t = 0 the self-similar coordinate is undefined; the exact initial
    piecewise-constant states are returned directly instead.
    """
    if t <= 0.0:
        left = x < 0.5
        rho = np.where(left, WL[0], WR[0])
        u = np.where(left, WL[1], WR[1])
        p = np.where(left, WL[2], WR[2])
        return np.stack([rho, u, p], axis=1)
    return np.array(
        [sample_riemann(WL, WR, GAMMA, (xi - 0.5) / t) for xi in x]
    )  # columns: (rho, u, p)


def run_sod(n=400):
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) / n
    U0 = conserved(*sod_initial(x), GAMMA)
    states = {}
    for order in (1, 2):
        states[order] = primitive(integrate(U0.copy(), T_SOD, dx, GAMMA, order=order), GAMMA)
    return x, states


def l1_errors(x, rho, u, p):
    ex = exact_profile(x, T_SOD)
    return (
        float(np.mean(np.abs(rho - ex[:, 0]))),
        float(np.mean(np.abs(u - ex[:, 1]))),
        float(np.mean(np.abs(p - ex[:, 2]))),
    )


def make_sod_cover(x, states):
    ex = exact_profile(x, T_SOD)
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for a, k, name, unit in ((ax[0], 0, "density ρ", "nondimensional"), (ax[1], 1, "velocity u", "nondimensional"), (ax[2], 2, "pressure p", "nondimensional")):
        a.plot(x, ex[:, k], "-", color=GRAY, lw=1.6, label="exact Riemann")
        a.plot(x, states[1][k], ".", ms=2.0, color=BLUE, label="first order (Rusanov)")
        a.plot(x, states[2][k], ".", ms=2.0, color=RED, label="MUSCL-SSP3", alpha=0.75)
        a.set_xlabel("x [nondimensional]")
        a.set_ylabel(f"{name} [{unit}]")
        a.set_title(name)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Sod shock tube at t = {T_SOD} (N=400, nondimensional) vs the exact Riemann solution",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "sod_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "sod_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def make_sod_gif(n=400, frames=41):
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) / n
    U0 = conserved(*sod_initial(x), GAMMA)

    # Record the second-order solution densely; exact profile drawn per frame.
    from finite_volume import step_muscl_ssp3

    U = U0.copy()
    t = 0.0
    dt_frame = T_SOD / (frames - 1)
    snapshots = [(0.0, primitive(U, GAMMA)[0].copy())]
    next_frame = dt_frame
    while t < T_SOD - 1e-14:
        dt_target = min(next_frame - t, T_SOD - t)
        U, dt = step_muscl_ssp3(U, dt_target, dx, GAMMA)
        t += dt
        if t >= next_frame - 1e-14:
            snapshots.append((t, primitive(U, GAMMA)[0].copy()))
            next_frame += dt_frame

    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=140)
    (line_num,) = ax.plot([], [], color=RED, lw=1.4, label="MUSCL-SSP3 (computed)")
    (line_ex,) = ax.plot([], [], color=GRAY, lw=1.4, ls="--", label="exact Riemann")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.05, 1.05)
    ax.set_xlabel("x [nondimensional]")
    ax.set_ylabel("density ρ [nondimensional]")
    title = ax.set_title("Sod shock tube — density at t = 0.000 s")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    writer = mpl_animation.PillowWriter(fps=8)
    with writer.saving(fig, RESULTS / "sod_evolution.gif", dpi=140):
        for t_s, rho in snapshots:
            ex = exact_profile(x, t_s)[:, 0]
            line_num.set_data(x, rho)
            line_ex.set_data(x, ex)
            title.set_text(f"Sod shock tube — density at t = {t_s:.3f} (nondimensional)")
            writer.grab_frame()
    plt.close(fig)
    return len(snapshots)


def run_contact_convergence():
    def initial(n):
        x = (np.arange(n) + 0.5) / n
        rho = 1.0 + 0.05 * np.exp(-((x - 0.5) / 0.08) ** 2)
        return x, conserved(rho, np.full(n, 0.5), np.full(n, 1.0), GAMMA)

    t_end = 0.1
    rows = ["n,order,L1_rho_error"]
    errors = {1: [], 2: []}
    for order in (1, 2):
        for n in (100, 200, 400, 800):
            x, U0 = initial(n)
            rho = primitive(integrate(U0, t_end, 1.0 / n, GAMMA, order=order), GAMMA)[0]
            exact = 1.0 + 0.05 * np.exp(-((x - 0.5 - 0.5 * t_end) / 0.08) ** 2)
            err = float(np.mean(np.abs(rho - exact)))
            errors[order].append(err)
            rows.append(f"{n},{order},{err:.12e}")
    (RESULTS / "contact_convergence.csv").write_text("\n".join(rows), encoding="utf-8")
    return {1: np.array(errors[1]), 2: np.array(errors[2])}


def make_convergence_figure(errors):
    ns = np.array([100, 200, 400, 800])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(ns, errors[1], "o-", color=BLUE, label="first order (Rusanov)")
    ax.loglog(ns, errors[2], "s-", color=RED, label="MUSCL-SSP3 (second order)")
    ax.loglog(ns, errors[1][0] * (ns / ns[0]) ** -1.0, "--", color=GRAY, lw=1.0, label=r"$\mathcal{O}(h)$")
    ax.loglog(ns, errors[2][0] * (ns / ns[0]) ** -2.0, ":", color=GRAY, lw=1.0, label=r"$\mathcal{O}(h^2)$")
    ax.set_xlabel("grid cells N [-]")
    ax.set_ylabel("L1 density error [-]")
    ax.set_title("Convergence on the exact advected contact\n(u = 0.5 passive wave, t = 0.1; nondimensional)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "contact_convergence.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "contact_convergence.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    x, states = run_sod()
    e1 = l1_errors(x, *states[1])
    e2 = l1_errors(x, *states[2])
    rows = ["scheme,L1_rho,L1_u,L1_p"]
    rows.append(f"first_order,{e1[0]:.12e},{e1[1]:.12e},{e1[2]:.12e}")
    rows.append(f"muscl_ssp3,{e2[0]:.12e},{e2[1]:.12e},{e2[2]:.12e}")
    (RESULTS / "sod_history.csv").write_text("\n".join(rows), encoding="utf-8")

    make_sod_cover(x, states)
    n_frames = make_sod_gif()

    errors = run_contact_convergence()
    make_convergence_figure(errors)

    o1 = np.log2(errors[1][:-1] / errors[1][1:])
    o2 = np.log2(errors[2][:-1] / errors[2][1:])
    p_star, u_star = riemann_exact(WL, WR, GAMMA)
    summary = {
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.platform(),
        },
        "sod": {
            "grid": 400,
            "t_end": T_SOD,
            "L1_rho_first_order": e1[0],
            "L1_rho_muscl_ssp3": e2[0],
            "exact_star_state": {"p_star": p_star, "u_star": u_star},
        },
        "contact_convergence": {
            "observed_orders_first": [float(v) for v in o1],
            "observed_orders_second": [float(v) for v in o2],
        },
        "animation": {"file": "sod_evolution.gif", "frames": n_frames},
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
