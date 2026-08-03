"""Experimental Hyperdimensional Space Computing kernels.

Modules in this package are research-only until their explicit promotion gate
is complete. Production trace retrieval continues to use its separately named
baseline engine.
"""

from ssa.hdsc.active_space import (
    MODEL_ID as H2_MODEL_ID,
)
from ssa.hdsc.active_space import (
    SUPPORT_METRIC,
    H2ActiveCluster,
    H2Audit,
    H2Candidate,
    H2Config,
    H2GainBounds,
    H2Result,
    H2State,
    bounded_active_shadow_step,
    state_total_variation,
)
from ssa.hdsc.directed_transport import (
    MODEL_ID as H1D_MODEL_ID,
)
from ssa.hdsc.directed_transport import (
    HDSCH1DAudit,
    HDSCH1DParameters,
    HDSCH1DResult,
    build_directed_rates,
    directed_generator,
    propagate_directed,
)
from ssa.hdsc.transport import (
    CLAIM_LEVEL,
    MICROSTATE_STATUS,
    MODEL_ID,
    MODELED_SCALE,
    HDSCH1Audit,
    HDSCH1Parameters,
    HDSCH1Result,
    HDSCInvariantError,
    allocate_source_mass,
    build_semantic_conductance,
    graph_laplacian,
    propagate,
    run_shadow,
)

__all__ = [
    "CLAIM_LEVEL",
    "H1D_MODEL_ID",
    "H2_MODEL_ID",
    "MICROSTATE_STATUS",
    "MODELED_SCALE",
    "MODEL_ID",
    "SUPPORT_METRIC",
    "H2ActiveCluster",
    "H2Audit",
    "H2Candidate",
    "H2Config",
    "H2GainBounds",
    "H2Result",
    "H2State",
    "HDSCH1Audit",
    "HDSCH1DAudit",
    "HDSCH1DParameters",
    "HDSCH1DResult",
    "HDSCH1Parameters",
    "HDSCH1Result",
    "HDSCInvariantError",
    "allocate_source_mass",
    "bounded_active_shadow_step",
    "build_directed_rates",
    "build_semantic_conductance",
    "directed_generator",
    "graph_laplacian",
    "propagate",
    "propagate_directed",
    "run_shadow",
    "state_total_variation",
]
