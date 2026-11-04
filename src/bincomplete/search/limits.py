"""Resolve proposal-budget terminology without changing generation semantics."""


def proposal_limit(max_checks, max_proposals, *, default):
    """Accept the explicit cap name and the stored compatibility keyword."""
    if max_checks is not None and max_proposals is not None and max_checks != max_proposals:
        raise ValueError("max_proposals and max_checks must agree when supplied together")
    return (
        default
        if max_checks is None and max_proposals is None
        else (max_proposals if max_proposals is not None else max_checks)
    )
