"""
Great Expectations suite definitions for FightIQ datasets.

Each suite describes the dataset category (raw/silver/gold), file name, and
expectations that need to be applied via GE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

DatasetCategory = Literal["raw", "silver", "gold"]


@dataclass(frozen=True)
class Expectation:
    expectation_type: str
    kwargs: Dict[str, Any]


@dataclass(frozen=True)
class DatasetSuite:
    name: str
    category: DatasetCategory
    filename: str
    expectations: List[Expectation]


SUITES: List[DatasetSuite] = [
    DatasetSuite(
        name="events_raw",
        category="raw",
        filename="events_raw.parquet",
        expectations=[
            Expectation("expect_table_columns_to_match_set", {"column_set": ["event_url", "event_date", "title"]}),
            Expectation("expect_column_values_to_not_be_null", {"column": "event_url"}),
            Expectation("expect_column_values_to_not_be_null", {"column": "event_date"}),
            Expectation("expect_column_values_to_match_regex", {"column": "event_url", "regex": r"^http"}),
            Expectation("expect_column_values_to_be_dateutil_parseable", {"column": "event_date"}),
        ],
    ),
    DatasetSuite(
        name="fights_raw",
        category="raw",
        filename="fights_raw.parquet",
        expectations=[
            Expectation("expect_table_columns_to_match_set", {"column_set": ["fight_url", "event_url"]}),
            Expectation("expect_column_values_to_not_be_null", {"column": "fight_url"}),
            Expectation("expect_column_values_to_not_be_null", {"column": "event_url"}),
            Expectation("expect_column_values_to_match_regex", {"column": "fight_url", "regex": r"^http"}),
            Expectation("expect_column_values_to_match_regex", {"column": "event_url", "regex": r"^http"}),
        ],
    ),
    DatasetSuite(
        name="fight_stats_raw",
        category="raw",
        filename="fight_stats_raw.parquet",
        expectations=[
            *[Expectation("expect_column_to_exist", {"column": col}) for col in [
                "fight_url",
                "fighter_name",
                "weight_class",
                "sig_strikes_succ",
                "sig_strikes_att",
                "total_strikes_succ",
                "total_strikes_att",
                "takedown_succ",
                "takedown_att",
                "submission_att",
                "reversals",
                "ctrl_time",
            ]],
            Expectation("expect_column_values_to_not_be_null", {"column": "fight_url"}),
            Expectation("expect_column_values_to_match_regex", {"column": "fight_url", "regex": r"^http"}),
            Expectation("expect_column_values_to_not_be_null", {"column": "fighter_name"}),
            Expectation("expect_column_values_to_be_between", {"column": "sig_strikes_succ", "min_value": 0}),
            Expectation("expect_column_values_to_be_between", {"column": "sig_strikes_att", "min_value": 0}),
            Expectation("expect_column_values_to_be_between", {"column": "total_strikes_succ", "min_value": 0}),
            Expectation("expect_column_values_to_be_between", {"column": "total_strikes_att", "min_value": 0}),
            Expectation("expect_column_values_to_be_between", {"column": "takedown_succ", "min_value": 0}),
            Expectation("expect_column_values_to_be_between", {"column": "takedown_att", "min_value": 0}),
        ],
    ),
    DatasetSuite(
        name="odds_raw",
        category="raw",
        filename="odds_raw.parquet",
        expectations=[
            *[Expectation("expect_column_to_exist", {"column": col}) for col in [
                "fight_url",
                "odds_f1",
                "odds_f2",
                "event_date",
                "source",
                "region",
            ]],
            Expectation("expect_column_values_to_not_be_null", {"column": "fight_url"}),
            Expectation("expect_column_values_to_match_regex", {"column": "fight_url", "regex": r"^http"}),
            Expectation("expect_column_values_to_be_between", {"column": "odds_f1", "min_value": 1.01, "max_value": 1000}),
            Expectation("expect_column_values_to_be_between", {"column": "odds_f2", "min_value": 1.01, "max_value": 1000}),
        ],
    ),
    DatasetSuite(
        name="rankings_raw",
        category="raw",
        filename="rankings_raw.parquet",
        expectations=[
            Expectation(
                "expect_table_columns_to_match_set",
                {"column_set": ["rank_date", "fighter", "weight_class", "rank"]},
            ),
            Expectation("expect_column_values_to_not_be_null", {"column": "rank_date"}),
            Expectation("expect_column_values_to_be_dateutil_parseable", {"column": "rank_date"}),
            Expectation("expect_column_values_to_not_be_null", {"column": "fighter"}),
            Expectation("expect_column_values_to_be_between", {"column": "rank", "min_value": 1, "max_value": 20}),
        ],
    ),
    DatasetSuite(
        name="fights_silver",
        category="silver",
        filename="fights_silver.parquet",
        expectations=[
            Expectation("expect_table_columns_to_match_set", {"column_set": ["fight_url", "event_url", "event_date", "event_name"]}),
            Expectation("expect_column_values_to_not_be_null", {"column": "fight_url"}),
            Expectation("expect_column_values_to_not_be_null", {"column": "event_date"}),
            Expectation("expect_column_values_to_match_regex", {"column": "fight_url", "regex": r"^http"}),
            Expectation("expect_column_values_to_match_regex", {"column": "event_url", "regex": r"^http"}),
        ],
    ),
    DatasetSuite(
        name="odds_silver",
        category="silver",
        filename="odds_silver.parquet",
        expectations=[
            *[Expectation("expect_column_to_exist", {"column": col}) for col in ["fight_url", "odds_f1", "odds_f2", "imp_f1_vigfree", "imp_f2_vigfree"]],
            Expectation("expect_column_values_to_not_be_null", {"column": "fight_url"}),
            Expectation("expect_column_values_to_be_between", {"column": "odds_f1", "min_value": 1.01, "max_value": 1000}),
            Expectation("expect_column_values_to_be_between", {"column": "odds_f2", "min_value": 1.01, "max_value": 1000}),
            Expectation("expect_column_values_to_be_between", {"column": "imp_f1_vigfree", "min_value": 0.0, "max_value": 1.0}),
            Expectation("expect_column_values_to_be_between", {"column": "imp_f2_vigfree", "min_value": 0.0, "max_value": 1.0}),
        ],
    ),
    DatasetSuite(
        name="rankings_silver",
        category="silver",
        filename="rankings_silver.parquet",
        expectations=[
            Expectation(
                "expect_table_columns_to_match_set",
                {"column_set": ["fight_url", "event_date", "fighter_name", "rank", "weight_class"]},
            ),
            Expectation("expect_column_values_to_not_be_null", {"column": "fight_url"}),
            Expectation("expect_column_values_to_not_be_null", {"column": "event_date"}),
            Expectation("expect_column_values_to_not_be_null", {"column": "fighter_name"}),
            Expectation("expect_column_values_to_be_between", {"column": "rank", "min_value": 1, "max_value": 20}),
        ],
    ),
    DatasetSuite(
        name="gold_features",
        category="gold",
        filename="gold_features.parquet",
        expectations=[
            *[Expectation("expect_column_to_exist", {"column": col}) for col in ["fight_url", "event_date", "f1_fighter_name", "f2_fighter_name"]],
            Expectation("expect_column_values_to_not_be_null", {"column": "fight_url"}),
            Expectation("expect_column_values_to_not_be_null", {"column": "event_date"}),
            Expectation("expect_column_values_to_be_unique", {"column": "fight_url"}),
        ],
    ),
]
