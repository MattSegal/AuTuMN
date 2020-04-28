from unittest import mock

import numpy as np
from sqlalchemy import create_engine


def get_mock_model(times, outputs, derived_outputs=None):
    mock_model = mock.Mock()
    mock_model.derived_outputs = derived_outputs or {}
    mock_model.outputs = np.array(outputs)
    mock_model.compartment_names = [
        "susceptibleXmood_happyXage_old",
        "susceptibleXmood_sadXage_old",
        "susceptibleXmood_happyXage_young",
        "susceptibleXmood_sadXage_young",
        "infectiousXmood_happyXage_old",
        "infectiousXmood_sadXage_old",
        "infectiousXmood_happyXage_young",
        "infectiousXmood_sadXage_young",
    ]
    mock_model.times = times
    mock_model.all_stratifications = {"mood": ["happy", "sad"], "age": ["old", "young"]}
    return mock_model


def in_memory_db_factory():
    """Replacement for _get_sql_engine, returns an in-memory database"""
    databases = {}

    def get_in_memory_db(db_path: str):
        # Create a fake DB path.
        with open(db_path, "w") as f:
            pass

        # Return an in-memory SQL Alchemy SQLite database engine.
        try:
            return databases[db_path]
        except KeyError:
            engine = create_engine("sqlite://", echo=False)
            databases[db_path] = engine
            return engine

    return get_in_memory_db
