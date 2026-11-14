"""BinComplete: membership-guided contig recruitment for bin completion.

The public API deliberately exposes data, models, proposals and structural
selections. The optional assessment CLI runs user-installed quality tools.
"""

from .data import Dataset, prepare_dataset
from .engine import ModelArtifact, ProposalArtifact, fit, propose, run
from .export import export
from .reconcile import SelectionArtifact, reconcile
from .resources import ResourceLimits
from .experiments import ExperimentArtifact, run_experiment
from .inventory import candidate_table

__version__ = "0.6.9b2"
__all__ = [
    "Dataset",
    "ModelArtifact",
    "ProposalArtifact",
    "SelectionArtifact",
    "ResourceLimits",
    "ExperimentArtifact",
    "run_experiment",
    "candidate_table",
    "prepare_dataset",
    "fit",
    "propose",
    "export",
    "reconcile",
    "run",
]
