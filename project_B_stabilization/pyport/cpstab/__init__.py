"""cpstab — Python port of the Shipley 2020 choroid-plexus z-stack
stabilization pipeline (LehtinenLab/Shipley2020, registration/).

Public API:

    from cpstab import RegistrationConfig, run_pipeline

    cfg = RegistrationConfig(input_path="/data/run002.sbx", refchannel=1)
    zproj = run_pipeline(cfg)

Ground truth for all numerics is the original MATLAB in
references/Shipley2020/; see README.md for the stage mapping, what was
dropped, and what is still unverified. Sibling registration modules
(orchestrator / apply_project / IO / writer) are resolved lazily by
pipeline.py, so importing this package works even while those modules are
still landing.
"""

from . import improved
from .config import RegistrationConfig, matlab_round
from .improved import (chain_refine_guard_scope, feature_scope, get_mode,
                       mode_scope, set_chain_refine_guard, set_mode)
from .pipeline import run_pipeline, matlab_uint16
from .precision import (compute_dtype_scope, get_compute_dtype,
                        set_compute_dtype)

__all__ = [
    "RegistrationConfig",
    "run_pipeline",
    "matlab_round",
    "matlab_uint16",
    # float32 fast mode (port extension; RegistrationConfig.compute_dtype is
    # the normal knob -- these are for callers that bypass run_pipeline, e.g.
    # fast_run.py's worker processes).
    "get_compute_dtype",
    "set_compute_dtype",
    "compute_dtype_scope",
    # 'improved' algorithm mode (port extension; RegistrationConfig.mode is
    # the normal knob). Same story as above for set_mode, plus feature_scope
    # for per-correction ablation -- see cpstab/improved.py.
    "improved",
    "get_mode",
    "set_mode",
    "mode_scope",
    "feature_scope",
    # correction 3's trust gate (RegistrationConfig.chain_refine_cap /
    # .chain_refine_min_ncc are the normal knobs; these are for the same
    # bypass-run_pipeline callers as set_mode).
    "set_chain_refine_guard",
    "chain_refine_guard_scope",
]

__version__ = "0.1.0"
