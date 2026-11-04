"""Compatibility names for previously serialized BinComplete model classes.

Use :mod:`bincomplete` for new code. These names resolve saved Python class
references without copying models, registering components again or refitting.
"""

from bincomplete import (
    Dataset,
    ExperimentArtifact,
    ModelArtifact,
    ProposalArtifact,
    ResourceLimits,
    SelectionArtifact,
    __version__,
    export,
    fit,
    prepare_dataset,
    propose,
    reconcile,
    run,
    run_experiment,
)
