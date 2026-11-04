"""Descriptive public component names and standard feature widths."""

from bincomplete import ResourceLimits
from bincomplete.core import build, registered
from bincomplete.methods import recipe
from bincomplete.standard import STANDARD_METHODS, standard_settings


def test_current_profiles_use_descriptive_feature_names_and_unchanged_widths():
    expected = {
        "composition": "composition_distances",
        "coverage": "coverage_agreement",
        "embedding": "embedding_similarities",
    }
    assert set(expected.values()) == set(registered("feature"))
    assert set(registered("classifier")) == {"hist_gradient_boosting", "scaled_logistic"}
    assert set(registered("positive_sampler")) == {"hidden_members"}
    for method in STANDARD_METHODS:
        spec = recipe(method, ResourceLimits(), standard_settings(method, device="cpu"))
        features = spec["features"]
        for key, component in features.items():
            assert component["name"] == expected[key]
        width = sum(len(build("feature", component).column_names()) for component in features.values())
        assert width == (0 if method in {"single", "cosine"} else 10 if method == "kmers" else 16)
