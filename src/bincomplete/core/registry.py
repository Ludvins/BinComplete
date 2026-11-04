import inspect

_REGISTRY = {}


def register(category, name):
    """Internal extension point: implement the protocol, then register a factory."""

    def decorate(factory):
        key = category, name
        if key in _REGISTRY:
            raise ValueError(f"Duplicate component: {category}.{name}")
        _REGISTRY[key] = factory
        return factory

    return decorate


def build(category, spec):
    key = category, spec["name"]
    if key not in _REGISTRY:
        raise ValueError(f"Unknown {category}: {spec['name']}; choices: {registered(category)}")
    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{category}.params must be an object")
    factory = _REGISTRY[key]
    try:
        inspect.signature(factory).bind(**params)
    except TypeError as error:
        raise ValueError(f"Invalid parameters for {category}.{spec['name']}: {error}") from error
    result = factory(**params)
    required = {
        "representation": ("identity", "transform"),
        "feature": ("column_names", "transform"),
        "masker": ("generate",),
        "positive_sampler": ("sample",),
        "negative_sampler": ("sample",),
        "retriever": ("eligible", "retrieve"),
        "classifier": ("fit", "predict_scores"),
        "search": ("propose",),
    }
    required["scorer"] = ("fit", "score")
    required["validation_strategy"] = ("evaluate",)
    if any(not callable(getattr(result, method, None)) for method in required.get(category, ())):
        raise ValueError(f"{category}.{spec['name']} does not implement its component interface")
    return result


def registered(category):
    return sorted(name for kind, name in _REGISTRY if kind == category)
