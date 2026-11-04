"""Representation dependencies shared by validation, hashing and lazy loading."""


def references(spec):
    if "representation" in spec and "representations" in spec:
        raise ValueError("Use representation or representations, not both")
    if "representations" in spec:
        mapping = spec["representations"]
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(not isinstance(v, str) for v in mapping.values())
        ):
            raise ValueError("representations must map input names to representation IDs")
        return mapping
    if "representation" in spec:
        if not isinstance(spec["representation"], str):
            raise ValueError("representation must be a representation ID")
        return {"value": spec["representation"]}
    return {}


def values(spec, getter, names=None):
    refs = references(spec)
    if names is not None:
        refs = {key: name for key, name in refs.items() if key in names}
    if "representation" in spec:
        return getter(spec["representation"])
    return {key: getter(name) for key, name in refs.items()} if refs else None
