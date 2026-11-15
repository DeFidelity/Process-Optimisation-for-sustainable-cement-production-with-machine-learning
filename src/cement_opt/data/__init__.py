"""Data loading, schema validation, and preprocessing utilities."""

from cement_opt.data.loader import load_and_merge_datasets
from cement_opt.data.preprocessing import (
    CONTROLLABLE_NUMERIC_FEATURES,
    FUEL_CATEGORIES,
    get_controllable_feature_names,
    prepare_modeling_dataframe,
)

__all__ = [
    "CONTROLLABLE_NUMERIC_FEATURES",
    "FUEL_CATEGORIES",
    "get_controllable_feature_names",
    "load_and_merge_datasets",
    "prepare_modeling_dataframe",
]
