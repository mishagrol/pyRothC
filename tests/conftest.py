from pathlib import Path

import pandas
import pytest


@pytest.fixture
def results_2years():
    with open(Path(__file__).parent.absolute() / "data" / "2years_results.csv", "r") as file:
        return pandas.read_csv(filepath_or_buffer=file, index_col=False)
