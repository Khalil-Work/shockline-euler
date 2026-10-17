"""finite_volume: 1D compressible Euler finite-volume solvers.

First-order Rusanov and second-order MUSCL(MC)-Rusanov schemes with an
exact Riemann solver used as the verification reference.
"""

from .euler1d import (
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

__all__ = [
    "conserved",
    "integrate",
    "primitive",
    "riemann_exact",
    "rusanov_flux",
    "sample_riemann",
    "sod_initial",
    "sound_speed",
    "step_muscl_ssp3",
    "step_rusanov",
]
__version__ = "1.0.0"
